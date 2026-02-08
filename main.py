import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import torch.distributed as dist
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, TrainingArguments, Trainer

from config import Config
from data import load_datasets
import deepspeed
deepspeed.init_distributed()

def load_model():
    print("Loading model...")
    # 加载处理器（Qwen3-VL 专用， 整合图像 + 文本处理）
    processor = AutoProcessor.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        dtype=Config.TORCH_DTYPE)

    # 加载 Tokenizer（设置 pad_token，企业级必备）
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer = tokenizer

    # 加载全精度模型（device_map="auto" + ZeRO-3自动分片）
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        Config.MODEL_NAME,
        TRUST_REMOTE_CODE=True,
        dtype=Config.TORCH_DTYPE,
        device_map = "auto",    # ZeRO-3 会覆盖此配置，自动分片到4卡
        attn_implementation="flash_attention_2",
        cache_dir="./model_cache"
    )

    # 为全精度训练准备模型（无需kbit预处理）
    # model = prepare_model_for_training(model)

    # 打印可训练参数（面试必提：仅 0.8%）
    model.print_trainable_parameters()
    return model, processor, tokenizer

def train():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank == 0:
        print("Rank 0: Downloading or loading model...")
        model, processor, tokenizer = load_model()
        train_ds, val_ds = load_datasets(processor)
        # 下载完成后，通知其他进程
        if world_size > 1:
            dist.barrier()  # rank 0 到达 barrier
    else:
        # 其他 rank 等待 rank 0 下载完成
        if world_size > 1:
            dist.barrier()
        print(f"Rank {local_rank}: Loading model from cache...")
        model, processor, tokenizer = load_model()
        train_ds, val_ds = load_datasets(processor)
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,                             # 模型 checkpoints、日志、TensorBoard 文件的保存目录
        per_device_train_batch_size=Config.BATCH_SIZE_PER_GPU,    # 每张 GPU 的 batch size。Qwen-VL-7B 显存大，常设为 1
        per_device_eval_batch_size=Config.BATCH_SIZE_PER_GPU,     # 验证时 batch size（可稍大，因无反向传播）
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,      # 梯度累积步数。等效于 global_batch_size = per_gpu_bs × grad_accum × num_gpus → 在小 batch 下模拟大 batch，稳定训练
        learning_rate=Config.LR,
        num_train_epochs=Config.EPOCHS,
        logging_steps=10,                                         # 每 10 步打印一次训练日志（loss、lr 等）
        save_strategy="epoch",                                    # 每个 epoch 结束时保存一次模型（也可设为 "steps"）
        evaluation_strategy="epoch",
        bf16=True,                                                # 全精度训练开启 bf16 混合精度
        deepspeed=Config.DS_CONFIG,
        save_total_limit=3,                                       # 最多保留最近 3 个 checkpoint，避免磁盘爆满
        load_best_model_at_end=True,                              # 训练结束后自动加载验证集上表现最好的模型（需配合 metric_for_best_model）
        metric_for_best_model="eval_loss",                        # 以验证损失（eval_loss）作为“最好模型”的评判标准
        greater_is_better=False,                                  # 表示指标越小越好（loss 越低越好）
        trust_remote_code=True,
        report_to="tensorboard",                                  # 将指标记录到 TensorBoard（也可选 "wandb", "mlflow"）
        local_rank=int(os.getenv('LOCAL_RANK', 0)),
        ddp_find_unused_parameters=False,                         # 在 DDP（Data Parallel）中跳过未使用参数的梯度同步
        gradient_checkpointing=True,                              # 配合 ZeRO-3 省显存 激活检查点（Activation Checkpointing） → 用时间换空间，显著降低显存（尤其对长序列/大模型）
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    if int(os.getenv('RANK', 0)) == 0:
        model.save_pretrained(Config.OUTPUT_DIR)
        processor.save_pretrained(Config.OUTPUT_DIR)

if __name__ == "__main__":
    train()