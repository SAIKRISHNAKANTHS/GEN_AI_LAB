from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

# ============================================================
# 1. Create Domain-Specific Dataset
# ============================================================

data = {
    "text": [
        "The transformer model achieved excellent accuracy.",
        "Large Language Models are revolutionizing AI.",
        "The football team won the championship.",
        "The cricket match was exciting.",
        "Neural networks are widely used in deep learning.",
        "The player scored a brilliant goal.",
        "Machine learning improves decision making.",
        "The tennis tournament starts tomorrow."
    ],

    "label": [
        1,  # Technology
        1,  # Technology
        0,  # Sports
        0,  # Sports
        1,  # Technology
        0,  # Sports
        1,  # Technology
        0   # Sports
    ]
}

# Convert dictionary into Hugging Face Dataset
dataset = Dataset.from_dict(data)

print("Dataset created successfully!")
print(dataset)


# ============================================================
# 2. Load BERT Tokenizer
# ============================================================

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


# ============================================================
# 3. Tokenization
# ============================================================

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


dataset = dataset.map(tokenize)

# Set dataset format for PyTorch
dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

print("\nTokenization completed!")


# ============================================================
# 4. Load Pretrained BERT Model
# ============================================================

print("\nLoading pretrained BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


# ============================================================
# 5. Training Configuration
# ============================================================

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",

    per_device_train_batch_size=2,

    num_train_epochs=2,

    logging_steps=1,

    save_strategy="no",

    report_to="none"
)


# ============================================================
# 6. Create Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)


# ============================================================
# 7. Fine-Tune BERT Model
# ============================================================

print("\nStarting BERT fine-tuning...")
print("=" * 60)

trainer.train()

print("\nTraining completed!")


# ============================================================
# 8. Save Fine-Tuned Model
# ============================================================

model_path = "./fine_tuned_model"

trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

print("\nFine-tuned model saved to:")
print(model_path)


# ============================================================
# 9. Load Fine-Tuned Model
# ============================================================

print("\nLoading fine-tuned model...")

classifier = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path
)


# ============================================================
# 10. Make Prediction
# ============================================================

text = "Generative AI models improve intelligent automation."

result = classifier(text)


# ============================================================
# 11. Convert Label to Class Name
# ============================================================

labels = {
    "LABEL_0": "Sports",
    "LABEL_1": "Technology"
}


# ============================================================
# 12. Display Prediction
# ============================================================

print("\n")
print("=" * 60)
print("PREDICTION")
print("=" * 60)

print("Input :", text)

print(
    "Predicted Class :",
    labels[result[0]["label"]]
)

print(
    "Confidence Score :",
    round(result[0]["score"], 3)
)

print("=" * 60)
