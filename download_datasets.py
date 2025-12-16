import os
import kaggle


def download_datasets():
    
    # Skip downloading dataset if it is already present
    if os.listdir("datasets/raw"):
        print("[INFO] Datasets already installed. Skipping...")
        return

    print("[INFO] Downloading datasets...")
    kaggle.api.dataset_download_files("doanquanvietnamca/liar-dataset", path="datasets/raw", unzip=True)


if __name__ == "__main__":
    download_datasets()
