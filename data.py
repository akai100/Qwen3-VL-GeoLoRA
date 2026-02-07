import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from datasets import load_dataset

def format_geometry3k_sample(example, processor, max_length=2048):
    print("Loading data...")
    texts = []
    images = []

    for q, a, img in zip(example["question"], example["answer"], example["image"]):
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text)
        images.append(img)

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    batch["labels"] = batch["input_ids"].clone()
    return batch

def load_datasets(processor):
    print("Loading Geometry3k dataset...")
    dataset = load_dataset("hiyouga/geometry3k", cache_dir="./data")

    print("Preprocessing train split...")
    train_ds = dataset["train"].map(
        lambda examples: format_geometry3k_sample(examples, processor),
        remove_columns=dataset["train"].column_names,
        desc="Formatting train samples"
    )

    print("Preprocessing validation split...")
    val_ds = dataset["validation"].map(
        format_geometry3k_sample,
        remove_columns=dataset["validation"].column_names,
        desc="Formatting validation samples"
    )