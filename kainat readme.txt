Chronic Kidney Disease (CKD) Classification — ML Baseline (Google Colab)
=======================================================================

Overview
--------
This repository contains a single Google Colab notebook that builds a baseline machine learning pipeline for **binary CKD classification**. It includes exploratory data analysis (EDA), feature scaling, model training, evaluation, feature selection, cross‑validation, and SHAP‑based explainability.

Notebook
--------
- **Kianat_project (1).ipynb** — end‑to‑end pipeline (run top‑to‑bottom in Google Colab).

Dataset
-------
- Expected input: a CSV file with a **`Class`** column (0 = Not CKD, 1 = CKD) and numeric feature columns.
- The notebook assumes an uploaded file named **`new_model.csv`** in `/content/` (Colab). You can either:
  1) Upload your CSV via the first cell (`files.upload()`) and **rename it to `new_model.csv`**, or
  2) Edit the path in the “Import Data” cell to point to your file (e.g., `/content/your_file.csv`).

Features referenced in the notebook (examples): `Hemo`, `Sg`, `Sc`, `Al`, `Htn`, `Bp`, `Bu`, `Sod`, `Pot`.  
> Note: The code expects features to already be numeric/encoded.

What the Notebook Does
----------------------
1. **EDA**
   - `df.info()`, missing‑value counts, correlation heatmap, class balance plot, and KDE plots for selected features.
2. **Preprocessing**
   - Standardization of continuous features using `StandardScaler` (e.g., `Bp`, `Bu`, `Sc`, `Sod`, `Pot`, `Hemo`).
3. **Modeling**
   - Trains baseline classifiers: **Gradient Boosting**, **SVM (probability=True)**, **k‑NN**, **MLP (Neural Network)**.
   - `train_test_split(test_size=0.20, random_state=42)`.
4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1‑score.
   - Plots: Confusion matrices and ROC curves (AUC on plot labels).
5. **Feature Selection**
   - `SelectKBest(mutual_info_classif)` sweep to see accuracy vs. number of features (with Gradient Boosting).
6. **Explainability**
   - **SHAP** values for the Gradient Boosting model, including summary plot.
7. **Cross‑Validation**
   - 5‑fold K‑Fold CV (`KFold(n_splits=5, shuffle=True, random_state=42)`) reporting mean ± std accuracy for each model.

Quick Start (Colab)
-------------------
1. Open the notebook in Google Colab.
2. Run the first cell to upload your CSV. Rename to **`new_model.csv`** (or update the path in the data‑load cell).
3. Run **Runtime → Run all** to execute the full pipeline.
4. Inspect printed metrics and generated plots at the end of each section.

Quick Start (Local)
-------------------
If you prefer local execution:
1. Create a virtual environment and install dependencies.
2. Open the notebook with Jupyter and update the CSV path to your local file.

Requirements
------------
The notebook is designed for Google Colab (most packages pre‑installed). If running locally, install:
- Python 3.10+
- `pandas >= 1.5`
- `numpy >= 1.23`
- `matplotlib >= 3.7`
- `seaborn >= 0.12`
- `scikit-learn >= 1.3`
- `shap >= 0.44`

Example (local):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

Customize / Extend
------------------
- **Models**: Edit the `models = { ... }` dictionary to add/remove classifiers (e.g., RandomForest, XGBoost).
- **CV**: Change `n_splits` or the scoring metric in the cross‑validation cell.
- **Scaling**: Adjust `continuous_features` to match your dataset.
- **Plots**: Add/remove figures as needed for your report.
- **Paths**: Update the CSV path in the Import Data cell to your file/location.

Reproducibility Notes
---------------------
- Random seeds are set with `random_state=42` where applicable.
- Fixed test split: `test_size=0.20`.
- CV uses `KFold` with `shuffle=True` and a fixed seed.

Outputs
-------
- Printed metrics per model (Accuracy, Precision, Recall, F1).
- Confusion matrix heatmaps.
- ROC curves with AUC per model.
- Feature selection results (k vs. accuracy).
- SHAP summary plot for Gradient Boosting.
- 5‑fold CV summary (mean ± std accuracy).

Repository Link (for FPR Front Page)
------------------------------------
Add your GitHub URL here (replace with your repo link):  
`https://github.com/<your-username>/<your-repo>`

License
-------
Add a license (e.g., MIT) if you intend to share/reuse the code.


Maintainer
----------
Author: <Your Name>  
Contact: <your.email@example.com>
