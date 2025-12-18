## Overview

This repository implements an end-to-end fake news detection system focused on political statements. The project leverages the **LIAR dataset** from PolitiFact and fine-tunes a **BERT-based transformer model** for binary classification of news statements.

Statements are classified as:

* **Label 1 (True):** true / mostly-true
* **Label 0 (False):** half-true / barely-true / false / pants-fire

The primary objective is to improve precision in identifying misleading or false information, making the system suitable for real-world misinformation mitigation scenarios.

## Project Structure

```
textfake-news-detection-project/
├── datasets/
│   ├── raw/                    # Downloaded raw TSV files
│   ├── train_preprocessed.csv
│   ├── valid_preprocessed.csv
│   └── test_preprocessed.csv
├── models/
│   ├── checkpoints/            # Training checkpoints
│   └── bert_finetuned/         # Final saved model and tokenizer
├── download_datasets.py        # Downloads LIAR dataset from Kaggle
├── preprocess_data.py          # Cleans text and binarizes labels
├── train_model.py              # Model preparation, training, and saving
├── main.py                     # Orchestrates the full pipeline
├── evaluate_model.py           # Optional evaluation and visualization
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---

## Requirements

* Python 3.8 or higher
* CUDA-capable GPU recommended (CPU fallback supported)

---

## Installation and Setup

1. **Clone the repository**

```bash
git clone https://github.com/timurenk0/predict-fake-news.git
cd predict-fake-news
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure Kaggle API credentials**

* Create a Kaggle account and generate an API token (`kaggle.json`).
* Place the file at:

  * Linux / macOS: `~/.kaggle/kaggle.json`
  * Windows: `%USERPROFILE%\.kaggle\kaggle.json`

---

## Usage

Run the complete pipeline (dataset download → preprocessing → training):

```bash
python main.py
```

You can also run predict.py separately to test model's capabilities. Input any statement from the test dataset and run script:
```bash
python predict.py
```

This process will:

* Download the LIAR dataset (if not already available)
* Preprocess and binarize the labels
* Fine-tune `bert-base-uncased` for 4 epochs
* Save the best-performing model (based on validation F1-score) to `./models/bert_finetuned`

---

## Future Work

* Extension to multi-class (6-label) truthfulness classification
* Fusion of textual features with metadata (speaker, party, context)
* Analysis of demographic vulnerability to misinformation
* User studies on trust in AI-based versus human fact-checkers
* Deployment as a REST API or browser extension

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Citation

If you use this repository or model in academic work, please cite:

