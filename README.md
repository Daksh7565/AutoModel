# AutoModel

A Streamlit app that trains machine learning models for you. Upload a dataset, pick the target column, and AutoModel cleans the data, compares several models, and gives you the best one as a downloadable file.

It supports three kinds of data:

| Mode | Input | Models |
|------|-------|--------|
| **Text** (tabular) | CSV file | Logistic Regression, SVC, Random Forest, XGBoost (classification) · Linear Regression, SVR, Random Forest, XGBoost (regression) |
| **NLP** | CSV with a text column and a label column | Logistic Regression, SVC, Random Forest, XGBoost on TF-IDF features |
| **Image** | Folder of images, one sub-folder per class | CNN, ResNet50, MobileNet (transfer learning) |

## Features

**Tabular data**
- **Dataset info:** row and column counts, and the column names.
- **Analysis:** finds out whether the target is for classification or regression, then checks for null values, text columns that need encoding, and class imbalance.
- **Recommended mode:** handles nulls (drops rows or fills them with the mean/median/mode), label-encodes text columns, balances classes with SMOTE, then runs 5-fold cross-validation on every candidate model and keeps the best one.
- **Manual mode:** you choose how nulls are handled, whether to apply SMOTE, the test-split size, and the model. You can also set hyperparameters.
- **Downloads:** the trained model (`best_model.sav`) and the cleaned dataset (`modified_dataset.csv`).

**NLP**: vectorises text with TF-IDF, compares classifiers, and saves the best one.

**Image**: splits the images 70/20/10 into train, validation, and test sets, applies augmentation, trains for 10 epochs, and shows the model summary, accuracy, and loss.

## Project structure

```
AutoModel/
├── streamlit.py          # Streamlit UI
├── MachineTraining.py    # Preprocessing, model selection and training logic
├── requirements.txt      # Python dependencies
└── Dataset/              # Sample datasets
    ├── data.csv                              # Boston housing (regression, target: MEDV)
    └── healthcare-dataset-stroke-data.csv    # Stroke prediction (classification, target: stroke)
```

## Setup

Requires Python 3.9–3.11 (TensorFlow does not support newer versions yet).

```powershell
cd C:\Users\Dell\Desktop\Project\AutoModel
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run streamlit.py
```

The app opens at http://localhost:8501.

### Quick try

1. In the sidebar, choose **Text**.
2. Upload `Dataset/healthcare-dataset-stroke-data.csv`.
3. Enter `stroke` as the output attribute.
4. Click **Analysis of dataset**. Then choose **Recommended** and click **Predictive system:**.

## Known limitations

- **Text mode is the web-ready feature.** Uploaded CSVs are saved into `Dataset/` (an upload with the same name as a sample file replaces it).
- **Image mode** reads a local folder of images and writes its splits to `C:\Image<project name>`, so it only works on Windows and not from a web deployment.
- **NLP mode** has no upload; it reads the file path exactly as typed. The **BOW** option currently uses TF-IDF as well.
- **The Hybrid image model** appears in the dropdown but is not implemented yet.
- **Manual-mode hyperparameters** are collected in the UI but not yet applied to the model.
- **Image models** are pickled to `model.sav` before training, so the saved file holds untrained weights.

## Tech stack

Python · Streamlit · pandas · scikit-learn · XGBoost · imbalanced-learn · TensorFlow/Keras · OpenCV · split-folders
