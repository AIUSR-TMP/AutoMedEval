import json

from torch import nn
from tqdm import tqdm

from Model_Training.flash_attention import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

import copy
import logging
import os
import wandb
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import Model_Training.utils as utils
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    set_peft_model_state_dict
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
    test_run_num: int = field(default=-1, metadata={"help": "Number of test runs."})
    prompt_template: str = field(default=None, metadata={"help": "Prompt template."})


@dataclass
class InferenceArguments:
    local_rank: int = field(default=-1, metadata={"help": "Local rank of the process."})
    wandb_project_name: str = field(default="Unknown", metadata={"help": "Name of the wandb project."})
    wandb_run_name: str = field(default="Unknown", metadata={"help": "Name of the wandb run."})
    output_dir: str = field(default="./Model_Training/Inference/Results",
                            metadata={"help": "Path to the output directory."})
    cache_dir: Optional[str] = field(default=None)
    max_length: int = field(default=512, metadata={"help": "Maximum generate sequence length"})
    temperature: float = field(default=0.4, metadata={"help": "Temperature for sampling."})
    top_p: float = field(default=0.7, metadata={"help": "Top p for sampling."})
    top_k: int = field(default=50, metadata={"help": "Top k for sampling."})
    per_device_eval_batch_size: int = field(default=4)
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample."})
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
    sources_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (sources, targets)]
    input_ids = sources_tokenized["input_ids"]
    labels = targets_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=labels)


class InferenceDataset(Dataset):
    """Dataset for inference."""

    def __init__(self, data_args: DataArguments, tokenizer: transformers.PreTrainedTokenizer):
        super(InferenceDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_args.data_path)
        if data_args.test_run_num > 0:
            list_data_dict = list_data_dict[-data_args.test_run_num:]
        logging.warning("Formatting inputs...")
        prompt = PROMPT_DICT[data_args.prompt_template]
        sources = [prompt.format_map(example) for example in list_data_dict]
        targets = []
        for example in list_data_dict:
            if "evaluation" in example:
                output = example["evaluation"]
            else:
                output = example["output"]
            targets.append(f"{output}{tokenizer.eos_token}")

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        logging.warning("Data tokenized completed.")

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def DataCollatorForInferenceDataset(instances: Sequence[Dict], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Collate examples for inference."""

    input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
    maxlength = max([input_id.__len__()] for input_id in input_ids)[0]
    input_ids = torch.stack([torch.cat(
        (torch.tensor([tokenizer.pad_token_id] * (maxlength - input_id.__len__()), dtype=torch.long), input_id.long()),
        0) for input_id in input_ids])
    maxlength = max([label.__len__()] for label in labels)[0]
    labels = torch.stack([torch.cat(
        (label.long(), torch.tensor([tokenizer.pad_token_id] * (maxlength - label.__len__()), dtype=torch.long)), 0) for
                          label in labels])
    return dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        labels=labels,
    )


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


def top_filtering(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    model_args, data_args, inference_args = parser.parse_args_into_dataclasses()

    if inference_args.local_rank <= 0:
        os.environ["WANDB_API_KEY"] = '33226dac1d1de682ed336dda35f04d7571d04b4d'
        run = wandb.init(project=inference_args.wandb_project_name, name=inference_args.wandb_run_name)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=inference_args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
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
    model.eval()
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=inference_args.cache_dir,
        model_max_length=inference_args.model_max_length,
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

    test_dataset = InferenceDataset(tokenizer=tokenizer, data_args=data_args, )
    data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=inference_args.per_device_eval_batch_size,
        collate_fn=lambda x: DataCollatorForInferenceDataset(x, tokenizer),
        shuffle=False
    )
    if model_args.lora_used:
        model_path = model_args.lora_ckpt
    else:
        model_path = model_args.model_name_or_path
    model_name = model_path.split("ckpt")[-1].strip('/').replace("/", "-")
    data_name = data_args.data_path.split("/")[-1].split(".")[0]
    output_file = os.path.join(
        inference_args.output_dir, f"{model_name}_{data_name}_{data_args.prompt_template}_{data_args.test_run_num}.json"
    )
    outputs = []
    text_table = wandb.Table(columns=["input", "output", "ground_truth"])
    for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            response_ids = model.generate(inputs=input_ids, attention_mask=attention_mask,
                                          max_new_tokens=inference_args.max_length,
                                          do_sample=inference_args.do_sample, top_k=inference_args.top_k,
                                          temperature=inference_args.temperature)
            input = tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
            response = tokenizer.batch_decode(
                response_ids, skip_special_tokens=True
            )
            ground_truth = tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
            for i in range(len(response)):
                text_table.add_data(input[i], response[i][len(input[i]):], ground_truth[i])
                outputs.append(
                    {
                        "input": input[i],
                        "output": response[i][len(input[i]):],
                        "ground_truth": ground_truth[i],
                    }
                )
    run.log({"example": text_table})
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    inference()
