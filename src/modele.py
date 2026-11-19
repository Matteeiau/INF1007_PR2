"""Régression linéaire « à la main » (NumPy), métriques et sauvegarde."""

from __future__ import annotations

import functools
import json
import os
import time
from typing import Callable

import joblib
import numpy as np

from src import exceptions as exc
from src.encodeur import Encodeur

NOM_MODELE = "modele.joblib"
NOM_METADONNEES = "metadonnees.json"


def chronometre(fonction: Callable) -> Callable:
    """Décorateur : mesure la durée d'exécution et la stocke dans self.duree_entrainement."""
    @functools.wraps(fonction)
    def enveloppe(self, *args, **kwargs):
        debut = time.perf_counter()
        resultat = fonction(self, *args, **kwargs)
        self.duree_entrainement = time.perf_counter() - debut
        return resultat
    return enveloppe


class RegressionLineaire:
    """Moindres carrés (Ridge si lambda_ > 0) résolus par X^T X beta = X^T y."""

    def __init__(self, lambda_: float = 0.0) -> None:
        """Crée un modèle non entraîné ; lambda_ >= 0 est la pénalité Ridge."""
        if lambda_ < 0:
            raise ValueError("lambda_ doit être positif ou nul.")
        self._lambda = lambda_
        self._beta: np.ndarray | None = None
        self.duree_entrainement: float | None = None

    @chronometre
    def entrainer(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calcule beta ; X doit contenir la colonne d'intercept en position 0."""
        A = X.T @ X
        if self._lambda > 0:
            penalite = self._lambda * np.eye(A.shape[0])
            penalite[0, 0] = 0.0  # on ne pénalise pas l'intercept
            A = A + penalite
        try:
            self._beta = np.linalg.solve(A, X.T @ y)
        except np.linalg.LinAlgError as erreur:
            raise exc.ErreurProjet(
                "Matrice X^T X singulière : augmentez LAMBDA dans config.py."
            ) from erreur

    def predire(self, X: np.ndarray) -> np.ndarray:
        """Renvoie X @ beta ; lève ModeleNonEntraineError si non entraîné."""
        if self._beta is None:
            raise exc.ModeleNonEntraineError()
        return X @ self._beta

    @property
    def beta(self) -> np.ndarray:
        """Copie des coefficients (lecture seule)."""
        if self._beta is None:
            raise exc.ModeleNonEntraineError()
        return self._beta.copy()

    def coefficients(self, noms: list[str]) -> dict[str, float]:
        """Associe chaque nom de colonne (Encodeur.noms_colonnes) à son coefficient."""
        beta = self.beta
        if len(noms) != len(beta):
            raise ValueError(f"{len(noms)} noms pour {len(beta)} coefficients.")
        return {nom: float(b) for nom, b in zip(noms, beta)}

    def __repr__(self) -> str:
        etat = f"{len(self._beta)} coefficients" if self._beta is not None else "non entraîné"
        return f"RegressionLineaire(lambda_={self._lambda}, {etat})"


# ---------------------------------------------------------------- métriques

def mae(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur absolue moyenne."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(y_pred))))


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Racine de l'erreur quadratique moyenne."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(y_pred)) ** 2)))


def r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de détermination R²."""
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        raise ValueError("R² indéfini : toutes les valeurs de y sont identiques.")
    return float(1 - ss_res / ss_tot)


def reference(y_train: np.ndarray, n: int) -> np.ndarray:
    """Modèle trivial : prédit toujours la moyenne de y_train."""
    return np.full(n, float(np.mean(y_train)))


# --------------------------------------------------------------- sauvegarde

def sauvegarder(
    modele: RegressionLineaire, encodeur: Encodeur, metriques: dict, dossier: str
) -> None:
    """Écrit modele.joblib et metadonnees.json dans dossier."""
    os.makedirs(dossier, exist_ok=True)
    joblib.dump(modele, os.path.join(dossier, NOM_MODELE))
    metadonnees = {
        **encodeur.vers_dict(),
        "metriques": {k: float(v) for k, v in metriques.items()},
    }
    with open(os.path.join(dossier, NOM_METADONNEES), "w", encoding="utf-8") as f:
        json.dump(metadonnees, f, indent=2, ensure_ascii=False)


def charger(dossier: str) -> tuple[RegressionLineaire, Encodeur, dict]:
    """Relit (modele, encodeur, metriques) ; FileNotFoundError si absents."""
    chemin_modele = os.path.join(dossier, NOM_MODELE)
    chemin_meta = os.path.join(dossier, NOM_METADONNEES)
    for chemin in (chemin_modele, chemin_meta):
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    modele = joblib.load(chemin_modele)
    with open(chemin_meta, encoding="utf-8") as f:
        metadonnees = json.load(f)
    return modele, Encodeur.depuis_dict(metadonnees), metadonnees["metriques"]