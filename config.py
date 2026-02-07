import torch
from peft import LoraConfig

class Config:
    MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
    DATA_PATH = "./data"
    OUTPUT_DIR = "./output"
    DS_CONFIG = "./ds_config_zero3.json"

    # 全精度配置（无量化）
    TORCH_DTYPE = torch.bfloat16 # bf16 比 fp32 节省 50% 显存，精度几乎无损失

    # LoRa 配置（全精度下 LoRa 参数占比仍仅 0.8%）
    LORA_CONFIG = LoraConfig(
        r = 32,
        lora_alpha = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout = 0.03,
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    # 训练配置（4卡总 batch_size = 16，适配全精度显存）
    BATCH_SIZE_PER_GPU = 2   # 全精度下单卡 batch 需降低
    GRAD_ACCUM_STEPS = 2
    LR = 1e-4
    EPOCHS = 5