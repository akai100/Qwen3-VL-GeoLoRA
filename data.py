import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from datasets import load_dataset

def format_geometry3k_sample(example, processor, max_visual_length=288, max_text_length=512):
    """
    修复：适配单样本处理（map函数设置batched=False），统一文本+图像维度
    """
    # 单样本处理（移除批量循环，适配map的单样本模式）
    try:
        # 打印当前处理的样本ID和进程ID，定位具体样本
        sample_id = example.get("id", "unknown")
        print(f"进程 {os.getpid()} 处理样本ID: {sample_id}")

        q = example["problem"]
        a = example["answer"]
        img = example["images"]
    
        # 构建Qwen3-VL的对话模板
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
    
        # 编码文本+图像（关键：单样本编码，避免批次维度混乱）
        # 先处理图像：确保图像是PIL对象且维度正确
        if isinstance(img, list):
            img = img[0] if len(img) > 0 else None  # 兼容数据集的图像格式
    
        encoding = processor(
            text=text,
            images=img,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length,  # 文本最大长度
            max_pixels=max_visual_length * 28 * 28,
        )

        # 3. 强制截断/填充pixel_values到288维（兜底保障）
        if "pixel_values" in encoding and encoding["pixel_values"].dim() > 1:
            pv = encoding["pixel_values"].squeeze(0)  # 移除batch维度
            # 截断过长的特征
            if pv.shape[1] > max_visual_length:
                pv = pv[:, :max_visual_length, :]
            # 填充过短的特征（用0填充）
            elif pv.shape[1] < max_visual_length:
                pad_len = max_visual_length - pv.shape[1]
                pv = torch.nn.functional.pad(pv, (0, 0, 0, pad_len), mode="constant", value=0)
            encoding["pixel_values"] = pv

    
        # 展平batch维度（单样本编码后会有(1, max_length)，需去掉batch维度）
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
    
        # 处理labels：padding部分设为-100（避免计算损失）
        encoding["labels"] = encoding["input_ids"].clone()
        padding_mask = encoding["input_ids"] == processor.tokenizer.pad_token_id
        encoding["labels"][padding_mask] = -100
    except Exception as e:
        # 捕获所有异常，避免子进程崩溃导致主进程卡死
        print(f"处理样本 {sample_id} 失败: {e}")
        traceback.print_exc()
        return None  # 返回空值，避免进程挂起
    print(f"进程 {os.getpid()} 完成样本ID: {sample_id}")    
    return encoding

def load_datasets(processor, max_length=2048):
    """
    修复：设置batched=False，单样本处理；添加数据集缓存和格式验证
    """
    # 加载数据集，指定缓存目录，避免重复下载
    dataset = load_dataset(
        "hiyouga/geometry3k", 
        cache_dir="./data",
        trust_remote_code=True
    )

    print("Preprocessing train split...")
    train_ds = dataset["train"].map(
        lambda example: format_geometry3k_sample(example, processor, max_visual_length=288, max_text_length=max_length),
        num_proc=1,
        batched=False,  # 关键：单样本处理，避免批次维度混乱
        remove_columns=dataset["train"].column_names,
        desc="Formatting train samples",
        #num_proc=os.cpu_count()  # 多进程加速预处理
    )
    print("train data process finished")

    print("Preprocessing validation split...")
    val_ds = dataset["validation"].map(
        lambda example: format_geometry3k_sample(example, processor, max_visual_length=288, max_text_length=max_length),
        batched=False,
        num_proc=1,
        remove_columns=dataset["validation"].column_names,
        desc="Formatting validation samples",
        #num_proc=os.cpu_count()
    )
    print("val data process finished")
    
    # 转换为PyTorch格式，避免后续DataLoader报错
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "labels"])
    
    return train_ds, val_ds