"""Point d'entrée du projet.

    python -m src.main entrainer     # entraîne, évalue, trace et sauvegarde
    python -m src.main               # lance l'interface graphique
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src import config as cfg
from src.donnees import (
    charger_csv,
    extraire_variables_multiples,
    nettoyer,
    separer,
    tracer_cible_par_categorie,
    tracer_distribution_cible,
)
from src.encodeur import Encodeur
from src.exceptions import ErreurProjet
from src.modele import RegressionLineaire, mae, r2, reference, sauvegarder


def preparer_donnees(chemin_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge, nettoie et sépare le CSV ; renvoie (df_propre, train, test)."""
    df = charger_csv(chemin_csv)
    df = extraire_variables_multiples(df, cfg.COLONNES_MULTIPLES)
    df = nettoyer(df, cfg.COLONNES_NUM, cfg.COLONNES_CAT, cfg.CIBLE)
    train, test = separer(df, cfg.PROPORTION_TEST, cfg.GRAINE)
    return df, train, test


def entrainer_et_evaluer(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[RegressionLineaire, Encodeur, dict[str, float]]:
    """Entraîne encodeur + régression sur train, évalue sur test contre la référence."""
    encodeur = Encodeur(cfg.COLONNES_NUM, cfg.COLONNES_CAT, cfg.MAX_CATEGORIES)
    encodeur.apprendre(train)
    X_train, X_test = encodeur.transformer(train), encodeur.transformer(test)
    y_train = train[cfg.CIBLE].to_numpy(dtype=float)
    y_test = test[cfg.CIBLE].to_numpy(dtype=float)

    modele = RegressionLineaire(getattr(cfg, "LAMBDA", 0.0))
    modele.entrainer(X_train, y_train)

    y_pred = modele.predire(X_test)
    y_ref = reference(y_train, len(y_test))
    metriques = {
        "mae_modele": mae(y_test, y_pred),
        "r2_modele": r2(y_test, y_pred),
        "mae_reference": mae(y_test, y_ref),
        "r2_reference": r2(y_test, y_ref),
        "n_train": len(train),
        "n_test": len(test),
    }
    return modele, encodeur, metriques


def formater_resultats(metriques: dict[str, float]) -> str:
    """Met les métriques en forme pour la console."""
    return (
        f"Train : {metriques['n_train']} jeux | Test : {metriques['n_test']} jeux\n"
        f"{'':12}{'MAE':>8}{'R²':>8}\n"
        f"{'Référence':12}{metriques['mae_reference']:8.2f}{metriques['r2_reference']:8.3f}\n"
        f"{'Régression':12}{metriques['mae_modele']:8.2f}{metriques['r2_modele']:8.3f}"
    )


def tracer_exploration(df: pd.DataFrame, dossier: str) -> None:
    """Enregistre les graphiques d'exploration dans le dossier donné."""
    tracer_distribution_cible(df, cfg.CIBLE, os.path.join(dossier, "distribution.png"))
    for c in cfg.COLONNES_CAT:
        tracer_cible_par_categorie(df, c, cfg.CIBLE, os.path.join(dossier, f"par_{c}.png"), 10)


def commande_entrainer(chemin_csv: str, dossier_modeles: str, dossier_figures: str) -> None:
    """Pipeline complet : données → graphiques → modèle → sauvegarde."""
    df, train, test = preparer_donnees(chemin_csv)
    tracer_exploration(df, dossier_figures)
    modele, encodeur, metriques = entrainer_et_evaluer(train, test)
    sauvegarder(modele, encodeur, metriques, dossier_modeles)
    print(formater_resultats(metriques))
    print(f"Modèle sauvegardé dans « {dossier_modeles} ».")


def commande_interface(dossier_modeles: str) -> None:
    """Lance l'interface (importée ici pour que l'entraînement n'en dépende pas)."""
    if not os.path.exists(os.path.join(dossier_modeles, "modele.joblib")):
        raise FileNotFoundError(
            "Aucun modèle sauvegardé : lancez d'abord « python -m src.main entrainer »."
        )
    from src.interface import lancer_interface
    lancer_interface(dossier_modeles)


def creer_parseur() -> argparse.ArgumentParser:
    """Définit les arguments de la ligne de commande."""
    parseur = argparse.ArgumentParser(description="Prédiction du metascore d'un jeu vidéo.")
    parseur.add_argument("commande", nargs="?", default="interface",
                         choices=["entrainer", "interface"])
    parseur.add_argument("--csv", default=cfg.CHEMIN_CSV)
    parseur.add_argument("--modeles", default=cfg.DOSSIER_MODELES)
    parseur.add_argument("--figures", default=cfg.DOSSIER_FIGURES)
    return parseur


def main(argv: list[str] | None = None) -> int:
    """Exécute la commande demandée ; renvoie 0 si succès, 1 si erreur prévue."""
    args = creer_parseur().parse_args(argv)
    try:
        if args.commande == "entrainer":
            commande_entrainer(args.csv, args.modeles, args.figures)
        else:
            commande_interface(args.modeles)
    except (ErreurProjet, FileNotFoundError, np.linalg.LinAlgError) as erreur:
        print(f"Erreur : {erreur}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())