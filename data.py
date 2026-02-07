import json
from datasets import load_dataset

def format_geometry3k_sample(example):
    user_prompt = (
        f"Promblem: {example['problem']}\n"
        f"Conclusion: {example['conclusion']}\n"
        "Please provide a step-by-step geometric proof."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        },
        {
            "role": "assistant",
            "content": example["proof"]
        }
    ]

    return {
        "image": [example['image']],
        "messages": messages
    }

def load_datasets():
    print("Loading Geometry3k dataset...")
    dataset = load_dataset("THUDM/geometry3k")

    print("Preprocessing train split...")
    train_ds = dataset["train"].map(
        format_geometry3k_sample,
        remove_columns=["pid", "problem", "conclusion", "proof", "image"],
        desc="Formatting train samples"
    )

    print("Preprocessing validation split...")
    val_ds = dataset["validation"].map(
        format_geometry3k_sample,
        remove_columns=["pid", "problem", "conclusion", "proof", "image"],
        desc="Formatting validation samples"
    )

    print(train_ds)
    print(val_ds)

load_datasets()