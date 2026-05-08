# Egyptian Currency Classification 🏦

A Convolutional Neural Network (CNN) that automatically classifies Egyptian pound banknotes from images using deep learning.

## Overview

This project uses a custom CNN built with TensorFlow/Keras to identify 9 classes of Egyptian currency notes, including both old and new designs.

**Target Accuracy:** ≥ 93% on the test set

## Classes

| Class | Description |
|-------|-------------|
| 1 EGP | One pound |
| 5 EGP | Five pounds |
| 10 EGP | Ten pounds |
| 10 EGP (new) | Ten pounds (new design) |
| 20 EGP | Twenty pounds |
| 20 EGP (new) | Twenty pounds (new design) |
| 50 EGP | Fifty pounds |
| 100 EGP | One hundred pounds |
| 200 EGP | Two hundred pounds |

## Dataset

- **Source:** [Egyptian Money Classification Dataset (2023 New Currency) — Kaggle](https://www.kaggle.com/datasets/belalsafy/egyptian-new-currency-2023/data)
- **Total Images:** 3,687
- **Split:**
  - Training: 2,637 images (71.5%)
  - Validation: 760 images (20.6%)
  - Test: 290 images (7.9%)

## Model Architecture

A custom CNN with 4 convolutional blocks followed by fully connected layers:

- **Block 1:** 2× Conv2D (32 filters) → BatchNorm → MaxPooling → Dropout
- **Block 2:** 2× Conv2D (64 filters) → BatchNorm → MaxPooling → Dropout
- **Block 3:** 2× Conv2D (128 filters) → BatchNorm → MaxPooling → Dropout
- **Block 4:** 2× Conv2D (256 filters) → BatchNorm → MaxPooling → Dropout
- **Classifier:** Flatten → Dense(256) → Dropout(0.3) → Dense(9, softmax)

**Training details:**
- Optimizer: Adam (lr=0.0003)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping + ReduceLROnPlateau
- Data Augmentation: rotation, zoom, shift, brightness variation

## Project Structure

```
├── Currencies.ipynb        # Main notebook (training, evaluation, prediction)
├── requirements.txt        # Python dependencies
└── dataset/                # Dataset folder (not included — download separately)
    ├── train/
    ├── valid/
    └── test/
```

## Setup & Usage

**1. Clone the repository**
```bash
git clone https://github.com/your-username/egyptian-currency-classification.git
cd egyptian-currency-classification
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download the dataset from Kaggle and place it in a `dataset/` folder with `train/`, `valid/`, and `test/` subfolders.

**4. Run the notebook**

Open `Currencies.ipynb` in Jupyter and run all cells. To predict on a new image, call:
```python
predict_image("path/to/your/image.jpg")
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- See `requirements.txt` for full list
