import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(model_path, test_path):
    # Load preprocessed dataset
    test_df = pd.read_csv(test_path)

    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Create prediction pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Create prediction logic
    predictions = classifier(test_df["statement"].tolist())
    probabilities = [p["score"] for p in predictions]
    preds = [1 if p > 0.5 else 0 for p in probabilities]
    labels = test_df["label"].values

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)

    # Print summary
    print("Model achievements summary:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:4f}")

    # Plot confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("graphs/confusion_matrix.png")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig("graphs/roc_curve.png")

    # Bar chart
    metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
    }

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color=["blue", "green", "orange", "red"])

    plt.title("Performance Metrics")
    plt.ylim(0, 1)
    plt.savefig("graphs/metrics_bar.png")

    print("Metrics hraphs saved in graphs directory")

if __name__ == "__main__":
    evaluate_model("models/bert_finetuned", "datasets/test_preprocessed.csv")
