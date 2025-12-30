# Preventive Detection of Chronic Diseases with AI – GitHub Readme

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)](https://jupyter.org/) [![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/) [![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-green?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/) [![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE) [![Paper](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge)](PARE1_2025-IRn°8_BAMOINtoure_MENGUEMELLAcolette_MBANGOBIANGdonia_NGUETTEFANEgad_YACKOUNDAMOUGOULAbelle.pdf)

**Paper**: [PARE1_2025-IR n°8 PDF](PARE1_2025-IRn°8_BAMOINtoure_MENGUEMELLAcolette_MBANGOBIANGdonia_NGUETTEFANEgad_YACKOUNDAMOUGOULAbelle.pdf)

## Project Overview
Empirical study behind the 2025 paper *“Preventive Detection of Chronic Diseases with Artificial Intelligence”*.  
We benchmark **KNN, MICE, median imputation** and **SMOTE** on four public clinical datasets to measure how missing-data handling affects **accuracy, recall, precision, F1** and **fairness across sex & age groups** for **Random-Forest & Dense Neural-Net** classifiers targeting hypertension and cardiovascular risk.

| Imputation | Stability | Fairness Tip | Best For |
|------------|-----------|--------------|----------|
| **MICE**   | Highest   | Keeps minority profiles | Correlated vars |
| **Median** | Medium    | Neutral/fast | Low-noise baseline |
| **KNN**    | Lowest    | Biased to majority | Homogeneous data |
| **SMOTE**  | –         | ↑ Recall on rare classes | Imbalanced targets |

&gt; “MICE emerges as the most stable approach … KNN tends to pull rare individuals toward the majority group, increasing false negatives.” — §5.1

## Datasets
| Dataset | Samples | Task | Missing % | Source |
|---------|---------|------|-----------|--------|
| NHANES | 5 704 | Hypertension | 12 % | [CDC](https://wwwn.cdc.gov/nchs/nhanes/) |
| Heart Disease | 303 | Diagnosis | 8 % | [Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) |
| Blood Pressure | 1 200 | Risk | 15 % | [Kaggle](https://www.kaggle.com/datasets/pavanbodanki/blood-pressure) |
| Hypertension Risk | 1 000 | Onset | 10 % | [Kaggle](https://www.kaggle.com/datasets/khan1803115/hypertension-riskmodel-main) |

## Quick Start, Structure, Citation & Authors
```bash
git clone https://github.com/your-username/Imputation-Fairness-Chronic-Diseases.git
cd Imputation-Fairness-Chronic-Diseases
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab
# run notebooks/01_preprocessing.ipynb → 02 → 03 → 04
```
Tree :
```
├── data/{raw,processed}
├── notebooks/
├── outputs/{tables,figures}
├── src/{imputers,metrics,viz}.py
└── README.md
```
BibTeX :
```
@techreport{PARE1_2025_IR8,
  title={Preventive Detection of Chronic Diseases with Artificial Intelligence},
  author={Bamoin, Toure Farouk and Mengué M'Ella, Colette and Mbango Biang, Donia and Nguette Fane, Gad and Yackounda Mougoula, Belle and Letard, Alexandre},
  institution={CERADE, ESAIP & LERIA, Université d'Angers},
  number={PARE1-2025-IR n°8},
  year={2025}
}
```
Authors (equal contribution) & Licence :  
Bamoin Toure Farouk, Mengué M’Ella Colette, Mbango Biang Donia, Nguette Fane Gad, Yackounda Mougoula Belle, Prof. Alexandre Letard — MIT © 2025 CERADE & LERIA
