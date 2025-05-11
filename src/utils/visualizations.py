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
        "Accuracy", "Recall", "Precision", "F1-score"
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


import plotly.graph_objects as go
from pathlib import Path


def plot_metric_by_dataset(all_metrics_df, metric_name, output_dir):
    """
    Génère un barplot par dataset pour une métrique donnée :
    - X : méthode d’imputation
    - Couleur : modèle
    """
    datasets = all_metrics_df["dataset"].unique()
    
    for dataset in datasets:
        df_subset = all_metrics_df[all_metrics_df["dataset"] == dataset]

        imputations = df_subset["imputation"].unique()
        models = ["random_forest", "dnn"]
        colors = {"random_forest": "#1f77b4", "dnn": "#ff7f0e"}  # bleu / orange

        fig = go.Figure()

        # Ajout des barres pour chaque modèle
        for model in models:
            values = []
            for method in imputations:
                value = df_subset[
                    (df_subset["imputation"] == method) &
                    (df_subset["modèle"] == model)
                ][metric_name].values
                values.append(value[0] if len(value) else None)

            fig.add_trace(go.Bar(
                x=imputations,
                y=values,
                name=model,
                marker_color=colors[model],
                text=[f"{v:.3f}" if v is not None else "" for v in values],
                textposition="auto"
            ))

        # Calcul et ajout des annotations de différence
        for method in imputations:
            try:
                val_rf = df_subset[
                    (df_subset["imputation"] == method) &
                    (df_subset["modèle"] == "random_forest")
                ][metric_name].values[0]

                val_dnn = df_subset[
                    (df_subset["imputation"] == method) &
                    (df_subset["modèle"] == "dnn")
                ][metric_name].values[0]

                delta = round(val_dnn - val_rf, 4)
                sign = "+" if delta >= 0 else ""

                y_max = max(val_rf, val_dnn)
                y_min = min(val_rf, val_dnn)
                y_mid = (y_max + y_min) / 2

                # Flèche et texte de différence
                fig.add_annotation(
                    x=method,
                    y=y_max + 0.03,
                    text=f"{sign}{delta:.4f}",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=1.2,
                    arrowcolor="black",
                    font=dict(color="red", size=13)
                )
            except:
                continue  # ignorer si une des valeurs est manquante
        
                # Titre et configuration de l'axe
        fig.update_layout(
            title={
                "text": f"{dataset.replace('_', ' ').capitalize()} – {metric_name.capitalize()}",
                "x": 0.5,
                "xanchor": "center",
                "font": dict(size=18)
            },
            barmode='group',
            xaxis=dict(
                title=dict(text="Méthode d’imputation", font=dict(size=14)),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title=dict(text=metric_name.capitalize(), font=dict(size=14)),
                tickfont=dict(size=12),
                range=[0, 1.05]
            ),
            legend=dict(title="Modèle", x=1.01, y=1, borderwidth=1),
            margin=dict(l=60, r=150, t=70, b=60)
        )

        # Export en PNG statique (nécessite kaleido installé)
        save_path = Path(output_dir) / f"{dataset}_{metric_name}_plotly_barplot.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(save_path), width=1000, height=600)
