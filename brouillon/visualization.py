import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_bar_metrics(df, save_path=None):
    """
    Génère un barplot comparatif des modèles selon les métriques principales.
    Entrée :
        df : DataFrame contenant les colonnes 'Imputation', 'Modèle', 'Accuracy', 'Recall', 'F1-score'
        save_path : chemin facultatif pour sauvegarder l'image
    """
    plt.figure(figsize=(10, 6))
    df_melted = df.melt(id_vars=["Imputation", "Modèle"],
                        value_vars=["Accuracy", "Recall", "F1-score"],
                        var_name="Métrique", value_name="Score")
    
    sns.barplot(data=df_melted, x="Métrique", y="Score",
                hue="Modèle", palette="Set2")
    plt.title("📊 Performance des modèles selon les métriques principales")
    plt.ylim(0, 1)
    plt.legend(title="Modèle")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_heatmap(df, save_path=None):
    """
    Affiche une heatmap des performances (modèle x imputation x métrique)
    """
    df_pivot = df.pivot(index="Imputation", columns="Modèle", values="F1-score")
    plt.figure(figsize=(8, 5))
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("🔥 Heatmap du F1-score par modèle et méthode d’imputation")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_radar(df, imputation=None, save_path=None):
    """
    Affiche un radar chart pour comparer plusieurs métriques sur un seul modèle ou une imputation donnée
    """
    from math import pi

    df_plot = df.copy()
    if imputation:
        df_plot = df_plot[df_plot["Imputation"] == imputation]

    metrics = ["Accuracy", "Recall", "Specificity", "F1-score", "AUC-ROC"]
    categories = metrics
    N = len(categories)

    plt.figure(figsize=(6, 6))
    for idx, row in df_plot.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]  # boucle pour fermer le graphe

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        plt.polar(angles, values, label=f"{row['Modèle']}", linewidth=2)
    
    plt.xticks(angles[:-1], categories)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.title(f"Radar des performances ({imputation})")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
