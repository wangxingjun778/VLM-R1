cd src/open-r1-multimodal

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Qwen2.5-VL-7B-GRPO-REC"
export LOG_PATH="./debug_log_$RUN_NAME.txt"


## Modified from the original script, origin:
# --nproc_per_node="8"
# --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
# --image_root <your_image_root>
# --num_generations 8

## Examples
# data: train2014/COCO_train2014_000000283267.jpg

## Records
# 1. Got OOM with 4 * A100-80G under num_generations=8, per_device_train_batch_size=1, gradient_accumulation_steps=2


## Notes
# Use latest version of trl: pip3 install "git+https://github.com/huggingface/trl.git@main"  (num_iterations set to 8)


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /mnt/nas2/xingjun.wxj/vlm_r1_work/models/Qwen2.5-VL-7B-Instruct \
    --dataset_name data_config/rec.yaml \
    --image_root /mnt/nas2/xingjun.wxj/vlm_r1_work/VLM-R1/data \
    --max_prompt_length 1024 \
    --num_generations 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
#    --resume_from_checkpoint /mnt/nas2/xingjun.wxj/vlm_r1_work/VLM-R1/src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-REC/checkpoint-xx

