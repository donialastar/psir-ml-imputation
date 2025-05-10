"""
visualizations.py
─────────────────
Génère des visualisations comparatives des performances modèles+imputations
à partir du CSV des métriques.

➡️ Utilisé à la fin du main.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_all_plots(metrics_df, output_dir):
    """
    Génère des barplots comparatifs pour les principales métriques.

    Paramètres :
        metrics_df : DataFrame
            DataFrame contenant les colonnes : Modèle, Imputation, Accuracy, Recall, F1-score, etc.
        output_dir : Path ou str
            Dossier où enregistrer les figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Colonnes numériques à tracer (peut être ajusté)
    metric_cols = [
        "Accuracy", "Balanced_Accuracy", "Recall", "Specificity",
        "Precision", "F1-score", "F2-score", "AUC-ROC", "AUC-PR", "MCC"
    ]

    # Concat Modèle + Imputation pour l'affichage
    metrics_df["Combinaison"] = metrics_df["Modèle"] + " + " + metrics_df["Imputation"]

    for metric in metric_cols:
        if metric not in metrics_df.columns:
            continue  # ignorer les colonnes manquantes

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=metrics_df.sort_values(by=metric, ascending=False),
            x="Combinaison",
            y=metric,
            palette="viridis"
        )
        plt.title(f"Comparaison de la métrique : {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Enregistrement
        fig_path = output_dir / f"{metric}_barplot.png"
        plt.savefig(fig_path)
        plt.close()
