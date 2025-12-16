from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


def _tokenize_function(batch, tokenizer):
    return tokenizer(
            batch["statement"],
            truncation=True,
            max_length=128,
            return_attention_mask=True
            )


def prepare_datasets(train_path, valid_path):
    # Load train and valid datasets
    raw_datasets = load_dataset("csv", data_files={"train": train_path, "valid": valid_path})

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Tokenize datasets
    tokenized_datasets = raw_datasets.map(
            lambda batch: _tokenize_function(batch, tokenizer),
            batched=True,
            remove_columns=["statement"]
    )

    tokenized_datasets = tokenized_datasets.with_format("torch")

    return tokenized_datasets, tokenizer


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.sigmoid(torch.tensor(logits)).squeeze()
    predictions = (probabilities > 0.5).int().numpy()
    labels = labels.astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", pos_label=1)

    return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
            }


def train_model(tokenized_datasets, tokenizer):
    # Initialize model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

    # Initialize data collator
    dc = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
            output_dir="./models/checkpoints",
            num_train_epochs=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            learning_rate=3e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to=[],
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            dataloader_num_workers=0
    )

    # Initialize model trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["valid"],
            tokenizer=tokenizer,
            data_collator=dc,
            compute_metrics=_compute_metrics
    )

    # Train the model
    trainer.train()

    save_dir = "./models/bert_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print("[SUCCESS] Training complete. Model save to the designated folder")
