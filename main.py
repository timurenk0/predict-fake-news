from download_datasets import download_datasets
from preprocess_data import preprocess_data
from train_model import prepare_datasets, train_model


if __name__ == "__main__":
    
    # Download datasets
    download_datasets()

    # Preprocess data
    preprocess_data("datasets/raw/test.tsv").to_csv("datasets/test_preprocessed.csv", index=False)
    preprocess_data("datasets/raw/train.tsv").to_csv("datasets/train_preprocessed.csv", index=False)
    preprocess_data("datasets/raw/valid.tsv").to_csv("datasets/valid_preprocessed.csv", index=False)

    # Train model
    tokenized_datasets, tokenizer = prepare_datasets("datasets/train_preprocessed.csv", "datasets/valid_preprocessed.csv")
    train_model(tokenized_datasets, tokenizer)
