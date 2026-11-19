"""Lecture, vérification et nettoyage du jeu de données."""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src import exceptions as exc


def charger_csv(chemin: str) -> pd.DataFrame:
    """Lit un fichier CSV et le renvoie sous forme de DataFrame.

    Args:
        chemin: chemin vers le fichier CSV.

    Returns:
        Le DataFrame lu.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        df = pd.read_csv(fichier)
    return df


def verifier_colonnes(df: pd.DataFrame, colonnes: list[str]) -> None:
    """Vérifie que toutes les colonnes demandées existent dans le DataFrame.

    Args:
        df: le DataFrame à vérifier.
        colonnes: noms des colonnes attendues.

    Raises:
        ColonneManquanteError: si au moins une colonne est absente.
    """
    colonnes_absentes = [col for col in colonnes if col not in df.columns]
    if colonnes_absentes:
        raise exc.ColonneManquanteError(colonnes_absentes)


def garder_colonnes(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Renvoie une copie du DataFrame réduite aux colonnes demandées.

    Args:
        df: le DataFrame d'origine (non modifié).
        colonnes: noms des colonnes à conserver.

    Returns:
        Un nouveau DataFrame ne contenant que ces colonnes.
    """
    return df[colonnes].copy()


def convertir_numeriques(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Convertit des colonnes en nombres ; les valeurs invalides deviennent NaN.

    Args:
        df: le DataFrame à convertir (modifié en place).
        colonnes: noms des colonnes à convertir.

    Returns:
        Le DataFrame converti.
    """
    for c in colonnes:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def nettoyer_categories(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Uniformise les colonnes catégorielles (texte, espaces, valeurs vides).

    Args:
        df: le DataFrame à nettoyer (modifié en place).
        colonnes: noms des colonnes catégorielles.

    Returns:
        Le DataFrame nettoyé ; les valeurs manquantes valent "Unknown".
    """
    for c in colonnes:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    return df


def supprimer_lignes_invalides(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Supprime les lignes incomplètes et les doublons.

    Args:
        df: le DataFrame à filtrer.
        colonnes: colonnes qui ne doivent pas contenir de NaN.

    Returns:
        Un nouveau DataFrame sans NaN dans ces colonnes ni doublons.
    """
    df = df.dropna(subset=colonnes)
    df = df.drop_duplicates()
    return df


def extraire_variables_multiples(df: pd.DataFrame, colonnes: dict[str, str]) -> pd.DataFrame:
    """Pour chaque colonne « A,B,C » du dict {source: nom}, crée nom = A et nb_<nom>s = 3.

    Les colonnes sources absentes du DataFrame sont ignorées.
    """
    df = df.copy()
    for source, nom in colonnes.items():
        if source not in df.columns:
            continue
        morceaux = df[source].astype("string").str.split(",")
        df[nom] = morceaux.str[0].str.strip()
        df[f"nb_{nom}s"] = morceaux.str.len()
    return df


def nettoyer(
    df: pd.DataFrame,
    colonnes_num: list[str],
    colonnes_cat: list[str],
    cible: str,
) -> pd.DataFrame:
    """Enchaîne toutes les étapes de nettoyage.

    Args:
        df: le DataFrame brut.
        colonnes_num: variables numériques d'entrée.
        colonnes_cat: variables catégorielles d'entrée.
        cible: nom de la colonne à prédire.

    Returns:
        Le DataFrame propre, prêt pour la séparation train/test.

    Raises:
        ColonneManquanteError: si une colonne demandée est absente.
    """
    verifier_colonnes(df, colonnes_num + colonnes_cat + [cible])
    df = garder_colonnes(df, colonnes_num + colonnes_cat + [cible])
    df = convertir_numeriques(df, colonnes_num + [cible])
    df = nettoyer_categories(df, colonnes_cat)
    df = supprimer_lignes_invalides(df, colonnes_num + [cible])
    return df


# ---------------------------------------------------------------------------
# Étape 5 : séparation entraînement / test
# ---------------------------------------------------------------------------

def separer(
    df: pd.DataFrame,
    proportion_test: float,
    graine: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare le jeu de données en ensembles d'entraînement et de test.

    La graine fixe le mélange : deux appels identiques donnent exactement
    le même découpage (résultats reproductibles).

    Args:
        df: le DataFrame nettoyé.
        proportion_test: part des lignes réservée au test, strictement entre 0 et 1.
        graine: graine aléatoire (random_state).

    Returns:
        Le couple (train, test).

    Raises:
        ValueError: si proportion_test n'est pas strictement entre 0 et 1.
    """
    if not 0 < proportion_test < 1:
        raise ValueError(
            f"proportion_test doit être entre 0 et 1 (reçu : {proportion_test})."
        )
    train, test = train_test_split(df, test_size=proportion_test, random_state=graine)
    return train, test


# ---------------------------------------------------------------------------
# Étape 6 : graphiques d'exploration
# ---------------------------------------------------------------------------

def _preparer_dossier(chemin_sortie: str) -> None:
    """Crée le dossier parent du fichier de sortie s'il n'existe pas."""
    dossier = os.path.dirname(chemin_sortie)
    if dossier:
        os.makedirs(dossier, exist_ok=True)


def tracer_distribution_cible(df: pd.DataFrame, cible: str, chemin_sortie: str) -> None:
    """Trace l'histogramme de la variable cible et l'enregistre en image.

    Args:
        df: le DataFrame nettoyé.
        cible: nom de la colonne à tracer.
        chemin_sortie: chemin du fichier image (ex. "figures/distribution.png").
    """
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


def tracer_cible_par_categorie(
    df: pd.DataFrame,
    colonne_cat: str,
    cible: str,
    chemin_sortie: str,
    max_categories: int = 10,
) -> None:
    """Trace un boxplot de la cible pour les catégories les plus fréquentes.

    Permet de voir, avant d'entraîner, si une variable catégorielle
    semble avoir un effet sur la cible.

    Args:
        df: le DataFrame nettoyé.
        colonne_cat: variable catégorielle (ex. "plateforme").
        cible: nom de la colonne cible.
        chemin_sortie: chemin du fichier image.
        max_categories: nombre de catégories affichées (les plus fréquentes).
    """
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