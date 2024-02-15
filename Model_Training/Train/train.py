from Model_Training.flash_attention import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

import json
import copy
import logging
import os
import wandb
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import Model_Training.utils as utils
import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)
from Model_Training.prompt import PROMPT_DICT

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lora_used: Optional[bool] = field(default=False)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)
    lora_ckpt: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_run_num: int = field(default=-1, metadata={"help": "Number of test runs."})
    prompt_template: str = field(default=None, metadata={"help": "Prompt template."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    local_rank: int = field(default=-1, metadata={"help": "Local rank of the process."})
    wandb_project_name: str = field(default="stanford_alpaca", metadata={"help": "Name of the wandb project."})
    wandb_run_name: str = field(default="Unknown", metadata={"help": "Name of the wandb run."})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args: DataArguments, tokenizer: transformers.PreTrainedTokenizer, train_args: TrainingArguments):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_args.data_path)
        if data_args.train_run_num > 0:
            list_data_dict = list_data_dict[:data_args.train_run_num]
        logging.warning("Formatting inputs...")
        prompt = PROMPT_DICT[data_args.prompt_template]
        sources = [
            PROMPT_DICT["prompt_suggestion_seperated"].format_map(example) if "suggestion1" in example else prompt.format_map(example)
            for example in list_data_dict
        ]
        targets = []
        for example in list_data_dict:
            if "evaluation" in example:
                output = example["evaluation"]
            else:
                output = example["output"]
            targets.append(f"{output}{tokenizer.eos_token}")
        if train_args.local_rank <= 0:
            print(sources[0])
            print(targets[0])
            print(sources[6])
            print(targets[6])
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        logging.warning("Data tokenized completed.")

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, train_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args, train_args=train_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_peft_config(peft_args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=peft_args.lora_rank,
        lora_alpha=32, lora_dropout=0.1
    )
    return peft_config


class MyTrainer(Trainer):

    def _save_checkpoint(self, _, trial, metrics=None):
        """ Don't save base model, optimizer etc. but create checkpoint folder (needed for saving adapter) """
        checkpoint_folder = f"{round(self.state.epoch)}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value

                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

        # save adapter config
        self.model.peft_config["default"].save_pretrained(output_dir)
        # get state dict through deepspeed engine
        engine_state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
        lora_state_dict = get_peft_model_state_dict(self.model, engine_state_dict)
        if self.args.local_rank == 0:
            torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
            print(f"Save adapter model at {output_dir}")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0:
        os.environ["WANDB_API_KEY"] = '33226dac1d1de682ed336dda35f04d7571d04b4d'
        wandb.init(project=training_args.wandb_project_name, name=training_args.wandb_run_name)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    if model_args.lora_used:
        print("Setup PEFT")
        peft_config = get_peft_config(peft_args=model_args)
        model = get_peft_model(model, peft_config)
        if model_args.lora_ckpt != "":
            checkpoint_name = os.path.join(
                model_args.lora_ckpt, "adapter_model.bin"
            )
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
        model.print_trainable_parameters()
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, train_args=training_args)
    if model_args.lora_used:
        trainer = MyTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
