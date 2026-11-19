"""Transformation d'un DataFrame en matrice X pour la régression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import exceptions as exc

AUTRE = "Autre"


class Encodeur:
    """Encode un DataFrame en X = [1 | numériques centrées | one-hot]."""

    def __init__(self, colonnes_num: list[str], colonnes_cat: list[str], max_categories: int = 40) -> None:
        """Crée un encodeur vide."""
        self._colonnes_num = list(colonnes_num)
        self._colonnes_cat = list(colonnes_cat)
        self._max_categories = max_categories
        self._moyennes: dict[str, float] = {}
        self._intervalles: dict[str, tuple[float, float]] = {}
        self._categories: dict[str, list[str]] = {}
        self._references: dict[str, str] = {}
        self._entraine = False

    def apprendre(self, train: pd.DataFrame) -> None:
        """Apprend moyennes, intervalles et catégories sur train uniquement."""
        for c in self._colonnes_num:
            self._moyennes[c] = float(train[c].mean())
            self._intervalles[c] = (float(train[c].min()), float(train[c].max()))

        for c in self._colonnes_cat:
            frequences = train[c].value_counts()
            principales = [str(v) for v in frequences.index[: self._max_categories]]
            if len(frequences) <= self._max_categories:
                self._references[c] = principales[0]
                self._categories[c] = principales[1:]
            else:
                self._references[c] = AUTRE
                self._categories[c] = principales

        self._entraine = True

    def _verifier_entraine(self) -> None:
        """Lève ErreurProjet si l'encodeur n'a pas été appris."""
        if not self._entraine:
            raise exc.ErreurProjet("L'encodeur doit être appris avant d'être utilisé.")

    def transformer(self, df: pd.DataFrame) -> np.ndarray:
        """Renvoie la matrice X ; une catégorie inconnue donne des 0."""
        self._verifier_entraine()
        colonnes = [np.ones(len(df))]
        for c in self._colonnes_num:
            colonnes.append(df[c].to_numpy(dtype=float) - self._moyennes[c])
        for c in self._colonnes_cat:
            valeurs = df[c].astype(str).to_numpy()
            for v in self._categories[c]:
                colonnes.append((valeurs == v).astype(float))
        return np.column_stack(colonnes)

    def noms_colonnes(self) -> list[str]:
        """Noms des colonnes de X, dans l'ordre de transformer()."""
        self._verifier_entraine()
        noms = ["intercept"] + self._colonnes_num
        for c in self._colonnes_cat:
            noms += [f"{c}={v}" for v in self._categories[c]]
        return noms

    def valeurs_possibles(self, colonne: str) -> list[str]:
        """Valeurs d'une catégorielle, référence en premier."""
        self._verifier_entraine()
        return [self._references[colonne]] + self._categories[colonne]

    def intervalle(self, colonne: str) -> tuple[float, float]:
        """(min, max) d'une numérique sur train."""
        self._verifier_entraine()
        return self._intervalles[colonne]

    @property
    def colonnes_num(self) -> list[str]:
        """Variables numériques (lecture seule)."""
        return list(self._colonnes_num)

    @property
    def colonnes_cat(self) -> list[str]:
        """Variables catégorielles (lecture seule)."""
        return list(self._colonnes_cat)

    def vers_dict(self) -> dict:
        """Exporte l'état appris en dict sérialisable JSON."""
        self._verifier_entraine()
        return {
            "colonnes_num": self._colonnes_num,
            "colonnes_cat": self._colonnes_cat,
            "max_categories": self._max_categories,
            "moyennes": self._moyennes,
            "intervalles": {c: list(v) for c, v in self._intervalles.items()},
            "categories": self._categories,
            "references": self._references,
        }

    @classmethod
    def depuis_dict(cls, d: dict) -> Encodeur:
        """Reconstruit un encodeur appris à partir de vers_dict()."""
        enc = cls(d["colonnes_num"], d["colonnes_cat"], d["max_categories"])
        enc._moyennes = dict(d["moyennes"])
        enc._intervalles = {c: tuple(v) for c, v in d["intervalles"].items()}
        enc._categories = {c: list(v) for c, v in d["categories"].items()}
        enc._references = dict(d["references"])
        enc._entraine = True
        return enc

    def __repr__(self) -> str:
        etat = "appris" if self._entraine else "non appris"
        return f"Encodeur(num={self._colonnes_num}, cat={self._colonnes_cat}, {etat})"