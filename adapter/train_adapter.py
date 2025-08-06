print("Starting protein property classification with LoRA adapter...")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import load_dataset
    from peft import get_peft_model, LoraConfig, TaskType
    import torch

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files="data/training_database.csv", delimiter=";")

    # Rename columns
    print("Renaming and selecting relevant columns...")
    dataset = dataset.rename_column("Amino_Acid_Sequence", "text")
    dataset = dataset.rename_column("Predicted_Stability", "label")
    
    keep_columns = ["text", "label"]
    dataset = dataset["train"].remove_columns([col for col in dataset["train"].column_names if col not in keep_columns])

    # Split dataset
    print("Splitting dataset...")
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Map labels to integers
    unique_labels = list(set(train_dataset["label"]))
    print("Unique labels:", unique_labels)
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

    def encode_labels(example):
        example["label"] = label_to_id[example["label"]]
        return example

    train_dataset = train_dataset.map(encode_labels)
    eval_dataset = eval_dataset.map(encode_labels)

    # Filter valid text entries
    def is_valid_text(example):
        return example["text"] and isinstance(example["text"], str)

    print("Cleaning dataset...")
    train_dataset = train_dataset.filter(is_valid_text)
    eval_dataset = eval_dataset.filter(is_valid_text)

    # Load tokenizer and base model
    print("Loading model and tokenizer...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_labels))

    print("Applying LoRA adapter...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    # Tokenize sequences
    print("Tokenizing sequences...")
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # Training setup
    training_args = TrainingArguments(
        output_dir="./results",
        do_eval=True,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer initialization
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    trainer.train()
    print("Training complete!")

except Exception as e:
    print(f"Error: {e}")