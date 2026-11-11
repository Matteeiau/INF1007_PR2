import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import src.exceptions as exc

def charger_csv(chemin: str) :
    if not os.path.exists(chemin): 
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    with open(chemin, encoding="utf-8") as f : 
        df = pd.read_csv(f)
    return df

def verifier_colonnes(df, colonnes):
    colonnes_absentes = [col for col in colonnes if col not in df.columns]
    if colonnes_absentes:
        raise exc.ColonneManquanteError(colonnes_absentes)
    return None

def garder_colonnes(df, colonnes) :
    return df[colonnes].copy()

def convertir_numeriques(df, colonnes) : 
    for c in colonnes : 
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def nettoyer_categories(df, colonnes) : 
    for c in colonnes : 
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    return df 

def supprimer_lignes_invalides(df, colonnes) : 
    df = df.dropna(subset=colonnes)
    df = df.drop_duplicates()
    return df 

def nettoyer(df, colonnes_num, colonnes_cat, cible):
    df = garder_colonnes(df, colonnes_num + colonnes_cat + [cible])
    df = convertir_numeriques(df, colonnes_num + [cible])
    df = nettoyer_categories(df, colonnes_cat)
    df = supprimer_lignes_invalides(df, colonnes_num + [cible])
    return df

def separer(df, proportion_test, graine) : 
    if not 0 < proportion_test < 1 : raise ValueError("proportion_test doit etre compris entre 0 et 1")
    train, test = train_test_split(df, test_size=proportion_test, random_state=graine) 
    return train, test

def _preparer_dossier(chemin_sortie) :
    dossier = os.path.dirname(chemin_sortie)
    if dossier:
        os.makedirs(dossier, exist_ok=True)
 
 
def tracer_distribution_cible(df, cible, chemin_sortie) :
    _preparer_dossier(chemin_sortie)
    valeurs = df[cible]
 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valeurs, bins=30, edgecolor="white")
    ax.axvline(valeurs.mean(), color="red", linestyle="--",
               label=f"Moyenne = {valeurs.mean():.1f}")
    ax.set_title(f"Distribution de « {cible} » ({len(valeurs)} jeux)")
    ax.set_xlabel(cible)
    ax.set_ylabel("Nombre de jeux")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin_sortie, dpi=150)
    plt.close(fig)
 
 
def tracer_cible_par_categorie(df, colonne_cat, cible, chemin_sortie, max_categories=10):
    _preparer_dossier(chemin_sortie)
    principales = df[colonne_cat].value_counts().index[:max_categories]
    groupes = [df.loc[df[colonne_cat] == cat, cible] for cat in principales]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(groupes, tick_labels=list(principales))
    ax.set_title(f"{cible} selon {colonne_cat} ({max_categories} plus fréquentes)")
    ax.set_xlabel(colonne_cat)
    ax.set_ylabel(cible)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(chemin_sortie, dpi=150)
    plt.close(fig)