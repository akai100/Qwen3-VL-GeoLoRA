import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import torch
import torch.distributed as dist
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model
from config import Config
from data import load_datasets
# 禁用 HuggingFace 自动设备分配（适配 DeepSpeed）
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 tokenizer 多线程冲突

def load_model():
    """加载 Qwen3-VL 模型 + Processor + Tokenizer（适配多卡缓存）"""
    print("Loading model...", flush=True)
    
    # 1. 加载处理器（Qwen3-VL 专用，整合图像+文本）
    processor = AutoProcessor.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        cache_dir="./model_cache"  # 统一缓存目录，避免多卡重复下载
    )

    # 2. 加载 Tokenizer（强制设置 pad_token）
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
        cache_dir="./model_cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 避免生成时截断错误
    processor.tokenizer = tokenizer

    # 3. 加载模型（适配 DeepSpeed ZeRO-3，关键配置）
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        Config.MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=Config.TORCH_DTYPE,
        device_map=None,  # 交给 DeepSpeed 分配
        low_cpu_mem_usage=True,  # 降低 CPU 内存占用
        cache_dir="./model_cache"
    )

    # 4. 显存优化配置
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}  # 适配 PyTorch 2.0+
    )
    torch.cuda.empty_cache()  # 清理显存

    # 5. 应用 LoRA 适配器
    model = get_peft_model(model, Config.LORA_CONFIG)
    model.print_trainable_parameters()  # 打印可训练参数占比

    return model, processor, tokenizer


class CustomVLDataCollator(DataCollatorForSeq2Seq):
    """
    自定义视觉-语言模型DataCollator，替代DataCollatorForVisionAndLanguageGeneration
    适配transformers旧版本，处理pixel_values和input_ids的统一padding
    """
    def __init__(self, processor, max_visual_length=288, max_text_length=512, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.max_visual_length = max_visual_length
        self.max_text_length = max_text_length

    def __call__(self, batch):
        # 1. 分离视觉特征（pixel_values）和文本特征
        pixel_values = []
        text_features = []
        for sample in batch:
            # 过滤空样本
            if sample is None:
                continue
            # 提取pixel_values并统一长度
            pv = sample.pop("pixel_values", None)
            if pv is not None:
                # 确保是tensor格式
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                # 截断/填充到max_visual_length（dim1）
                if pv.dim() == 3:  # 格式：[channels, length, hidden_dim]
                    if pv.shape[1] > self.max_visual_length:
                        pv = pv[:, :self.max_visual_length, :]
                    elif pv.shape[1] < self.max_visual_length:
                        pad_len = self.max_visual_length - pv.shape[1]
                        pv = torch.nn.functional.pad(pv, (0, 0, 0, pad_len), value=0)
                pixel_values.append(pv)
            # 提取文本特征（input_ids/attention_mask等）
            text_features.append(sample)

        # 2. 处理文本特征的padding（复用Seq2Seq的collate逻辑）
        text_batch = super().__call__(text_features)

        # 3. 合并视觉特征到批次中
        if pixel_values:
            # 堆叠所有pixel_values为批次张量
            text_batch["pixel_values"] = torch.stack(pixel_values)

        return text_batch

def get_data_collator(processor, max_visual_length=288, max_text_length=512):
    """创建适配Qwen3-VL的DataCollator，自动处理批次padding/truncation"""
    data_collator = CustomVLDataCollator(
        processor=processor,
        max_visual_length=max_visual_length,
        max_text_length=max_text_length,
        tokenizer=processor.tokenizer,  # 传给父类的tokenizer
        padding="max_length",
        max_length=max_text_length,
        return_tensors="pt",
    )
    # 自定义修正pixel_values的padding逻辑（兜底）
    original_call = data_collator.__call__
    def custom_collate(batch):
        # 第一步：过滤空样本
        batch = [x for x in batch if x is not None]
        # 第二步：统一pixel_values长度
        for sample in batch:
            if "pixel_values" in sample:
                pv = sample["pixel_values"]
                if pv.shape[1] != max_visual_length:
                    # 截断/填充到288维
                    if pv.shape[1] > max_visual_length:
                        sample["pixel_values"] = pv[:, :max_visual_length, :]
                    else:
                        pad_len = max_visual_length - pv.shape[1]
                        sample["pixel_values"] = torch.nn.functional.pad(
                            pv, (0, 0, 0, pad_len), mode="constant", value=0
                        )
        # 第三步：调用原始collate逻辑
        return original_call(batch)
    data_collator.__call__ = custom_collate
    return data_collator

def train():
    """核心训练函数（修复分布式同步 + 异常处理 + 资源优化）"""
    # ========== 1. 初始化分布式环境 ==========
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist_initialized = False  # 标记分布式是否初始化成功

    try:
        # 多卡训练初始化分布式
        #if False:
            #torch.cuda.set_device(local_rank)  # 绑定当前进程到对应 GPU
            #dist.init_process_group(
            #    backend="nccl",  # GPU 通信必须用 NCCL
            #    init_method="env://",
            #    world_size=world_size,
            #    rank=local_rank
            #)
            #dist_initialized = True
            #print(f"Rank {local_rank}: Distributed environment initialized", flush=True)

        # ========== 2. 同步加载模型和数据集 ==========
        model, processor, tokenizer = None, None, None
        train_ds, val_ds = None, None

        # Rank 0 先加载（避免多卡重复下载）
        if True:
            print("Rank 0: Downloading/loading model and dataset...", flush=True)
            model, processor, tokenizer = load_model()
            train_ds, val_ds = load_datasets(processor)
            print("Rank 0: Load completed, notifying other ranks...", flush=True)

        # 所有 rank 同步（仅分布式初始化成功时执行）
        #if dist_initialized and dist.is_initialized():
            #dist.barrier()

        # 非 0 rank 从缓存加载
        # if False:
        #    print(f"Rank {local_rank}: Loading from cache...", flush=True)
        #    model, processor, tokenizer = load_model()
        #    train_ds, val_ds = load_datasets(processor)

        # 加载完成后再次同步
        # if dist_initialized and dist.is_initialized():
        #    dist.barrier()

        # ========== 4. 配置训练参数（修复 DeepSpeed 冲突） ==========
        training_args = TrainingArguments(
            output_dir=Config.OUTPUT_DIR,
            per_device_train_batch_size=Config.BATCH_SIZE_PER_GPU,
            per_device_eval_batch_size=Config.BATCH_SIZE_PER_GPU,
            gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
            learning_rate=Config.LR,
            num_train_epochs=Config.EPOCHS,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=True,  # 适配 Ampere 架构 GPU（V100/A100）
            deepspeed=Config.DS_CONFIG,  # DeepSpeed 配置文件路径
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            # 移除重复配置，交给 DeepSpeed 自动处理
            ddp_find_unused_parameters=False,
            weight_decay=0.01,
            # 显存优化
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        # ========== 5. 多模态兼容的 DataCollator ==========
        data_collator = get_data_collator(processor, max_visual_length=288)

        # ========== 6. 初始化 Trainer ==========
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
        )

        # ========== 7. 启动训练 ==========
        trainer.train()

        # ========== 8. 仅主进程保存模型 ==========
        if local_rank == 0:
            model.save_pretrained(Config.OUTPUT_DIR)
            processor.save_pretrained(Config.OUTPUT_DIR)
            print(f"Model saved to {Config.OUTPUT_DIR}", flush=True)

    except Exception as e:
        # 异常处理：避免多卡卡死，快速清理资源
        print(f"Rank {local_rank} training error: {str(e)}", flush=True)
        # 销毁分布式进程组
        if dist_initialized and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except:
                pass
        # 清理显存
        torch.cuda.empty_cache()
        raise e  # 抛出异常，终止进程

    finally:
        # 最终清理：销毁分布式环境 + 释放显存
        if dist_initialized and dist.is_initialized():
            try:
                dist.destroy_process_group()
                print(f"Rank {local_rank}: Distributed process group destroyed", flush=True)
            except Exception as e:
                print(f"Rank {local_rank}: Destroy process group failed: {str(e)}", flush=True)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 强制设置 CUDA 可见设备（可选，按实际 GPU 数量调整）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 仅使用 0、1 号卡
    train()