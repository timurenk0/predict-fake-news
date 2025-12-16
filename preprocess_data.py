import pandas as pd


def preprocess_data(file_path):
    # Load data from the dataset
    df = pd.read_csv(file_path, sep="\t", header=None)

    # Assign column names based on the dataset schema
    df.columns = ["id", "label", "statement", "subject", "speaker", "job", "state", "party", "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context"]

    # Sanitize statement column values for easier interpretation and less confusion
    df["statement"] = df["statement"].str.replace(r"[^\w\s]", "", regex=True).str.lower()
    
    # Binarize label column values for the model training
    df["label"] = df["label"].map({
        "true": 1, "mostly_true": 1,
        "half_true": 0, "barely_true": 0, "false": 0, "pants_fire": 0
        })

    # Drop null values in statement and label columns
    df = df.dropna(subset=["statement", "label"])

    return df


if __name__ == "__main__":
    train_df = preprocess_data("datasets/raw/train.tsv")
    train_df.to_csv("datasets/train_processed.csv")
