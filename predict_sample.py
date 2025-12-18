from transformers import BertTokenizerFast, BertForSequenceClassification, pipeline


def predict_sample(model_path, sample):
    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=1)

    # Create pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Predict sample credibility
    result = classifier(sample)[0]
    probability = result["score"]
    label = 1 if probability > 0.5 else 0
    
    return label, probability


if __name__ == "__main__":
    sample = "mccain opposed a requirement that the government buy americanmade motorcycles and he said all buyamerican provisions were quote disgraceful"
    label, probability = predict_sample("models/bert_finetuned", sample)
    print(f"Label: {label} (1=credible, 0=not)\nProbability: {probability:.4f}")
