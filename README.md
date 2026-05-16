# Privacy-Preserving Network Security
## Classifying Encrypted Malicious Traffic using Ensemble ML

**Authors:** Moeez Jamal (22i0513) · Ammad Nasir (22i0477)  
**Course:** Computer Networks — Semester Project Phase 2  
**Dataset:** CICIDS2017

---

## Verified Results on Real CICIDS2017 (15% sample, 273k rows after dedup)

| Model | Accuracy | F1 Macro | ROC-AUC | **PR-AUC ★** |
|---|---|---|---|---|
| **XGBoost** | **99.91%** | **99.84%** | 1.0000 | **1.0000** |
| LightGBM | 99.90% | 99.83% | 1.0000 | 1.0000 |
| Random Forest | 99.90% | 99.82% | 0.9999 | 0.9998 |

**Total runtime**: ~11 minutes end-to-end on a standard laptop CPU.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download real CICIDS2017 data (~810 MB, takes ~90 sec on HF mirror)
python download_data.py

# 3. Open the notebook
jupyter notebook pipeline.ipynb

# 4. Run All Cells
```

The notebook is preconfigured for **real CICIDS2017 data**. To run a quick demo
without downloading anything, set `DEMO_MODE = True` in the Configuration cell.

---

## Project Structure

```
cnet proj/
├── pipeline.ipynb          ← Main notebook (run this)
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py      ← CICIDS2017 CSV loader
│   ├── demo_data.py        ← Synthetic data generator
│   ├── preprocessor.py     ← Cleaning, scaling, encoding
│   ├── resampling.py       ← SMOTE, Tomek, SMOTETomek
│   ├── models.py           ← XGBoost, RF, LightGBM, Ensemble
│   └── evaluation.py       ← PR-AUC, ROC-AUC, plots
├── data/                   ← Place real CICIDS2017 CSVs here
└── outputs/
    ├── models/             ← Saved .pkl model files
    ├── plots/              ← All generated PNG plots
    └── results_summary.json
```

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| **Load** | Merges all CICIDS2017 CSV files, normalises column names, cleans labels |
| **EDA** | Class distribution, feature distributions, correlation heatmap |
| **Preprocess** | Drop identifiers, impute Inf/NaN with median, remove constant cols, encode labels |
| **Split** | Stratified 80/20 train-test split + RobustScaler |
| **Resample** | SMOTETomek (SMOTE over-sample + Tomek Links boundary cleaning) |
| **Train** | XGBoost (cost-sensitive), Random Forest (balanced), LightGBM (balanced) |
| **Evaluate** | PR-AUC ★, ROC-AUC, F1 macro/weighted, Confusion Matrix |
| **Compare** | Side-by-side bar charts + resampling ablation study |
| **Save** | All models + scaler + label encoder + JSON summary |

---

## Key Design Decisions

### Why PR-AUC over accuracy?
CICIDS2017 is ~83% BENIGN. A model predicting "always BENIGN" gets 83% accuracy but 0% attack detection. PR-AUC directly measures minority-class (attack) precision vs recall.

### Why SMOTETomek?
- **SMOTE** generates synthetic minority samples via k-NN interpolation — avoids info loss from under-sampling
- **Tomek Links** removes borderline majority samples that could confuse the decision boundary
- Together they both expand and clean the training distribution

### Why XGBoost as primary model?
- `scale_pos_weight` parameter directly implements cost-sensitive learning for imbalanced classes
- Gradient boosting iteratively focuses on misclassified (minority) samples
- `eval_metric="aucpr"` optimises PR-AUC during training

---

## Using the Real CICIDS2017 Dataset

1. Download `MachineLearningCSV.zip` from [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Extract all `.csv` files into the `data/` folder
3. Set `DEMO_MODE = False` in the notebook Configuration cell
4. Re-run all cells

---

## Configuration Options

```python
DEMO_MODE      = True           # False = load real CSVs from data/
DEMO_SCALE     = 1.0            # 0.5 = half-size dataset (faster)
BINARY         = True           # False = 15-class multiclass
RESAMPLE_STRAT = "smote_tomek"  # none | smote | tomek | smote_tomek | smote_enn
PRIMARY_MODEL  = "xgboost"      # xgboost | random_forest | lightgbm | ensemble
TEST_SIZE      = 0.20
```
