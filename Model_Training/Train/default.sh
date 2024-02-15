# export WANDB_MODE="offline"
working_path=""
export PYTHONPATH=$working_path

cd $working_path
conda init
source activate
conda activate Evaluator
conda info --envs

deepspeed --master_port 12458 --num_gpus=8 Model_Training/Train/train.py \
    --model_name_or_path ./Model_Training/Model/medllama-13b/ \
    --data_path ./Model_Training/Dataset/test.json \
    --wandb_project_name "Unknown" \
    --wandb_run_name "Unknown" \
    --model_max_length 2048 \
    --output_dir ./Model_Training/Train/ckpt/test \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --deepspeed ./Model_Training/Train/configs/ds_config_zero2_offload_opt.json \
