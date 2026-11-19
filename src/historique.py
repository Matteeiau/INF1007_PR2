"""Historique des prédictions, stocké dans un fichier CSV."""

from __future__ import annotations

import csv
import os
from datetime import datetime

from src.exceptions import ErreurProjet


def _formater(valeur: object) -> object:
    """Affiche 2.0 comme 2 et laisse le reste intact."""
    if isinstance(valeur, float) and valeur.is_integer():
        return int(valeur)
    return valeur


class Historique:
    """Ajoute chaque prédiction à un fichier CSV et permet de la relire."""

    def __init__(self, chemin: str, colonnes: list[str]) -> None:
        """Prépare l'historique (le fichier n'est créé qu'au premier ajout).

        Args:
            chemin: chemin du fichier CSV.
            colonnes: noms des variables d'entrée, dans l'ordre d'affichage.
        """
        self.chemin = chemin
        self.entrees = list(colonnes)
        # colonnes du fichier : les variables, puis la prédiction et la date
        self.colonnes = self.entrees + ["prediction", "date"]

    def ajouter(self, valeurs: dict, prediction: float) -> None:
        """Ajoute une ligne à la fin du fichier (créé avec son en-tête si besoin).

        Args:
            valeurs: valeurs saisies, indexées par nom de variable.
            prediction: note prédite.

        Raises:
            KeyError: si une variable manque dans ``valeurs``.
            ErreurProjet: si le fichier existe déjà avec d'autres colonnes.
            OSError: si le fichier ne peut pas être écrit.
        """
        ligne = [_formater(valeurs[nom]) for nom in self.entrees]
        ligne.append(round(prediction, 1))
        ligne.append(datetime.now().strftime("%Y-%m-%d %H:%M"))

        nouveau = not os.path.exists(self.chemin) or os.path.getsize(self.chemin) == 0
        if not nouveau:
            self._verifier_en_tete()
        else:
            dossier = os.path.dirname(self.chemin)
            if dossier:
                os.makedirs(dossier, exist_ok=True)

        with open(self.chemin, "a", newline="", encoding="utf-8") as fichier:
            ecrivain = csv.writer(fichier)
            if nouveau:
                ecrivain.writerow(self.colonnes)
            ecrivain.writerow(ligne)

    def lire(self) -> list[dict]:
        """Relit toutes les lignes (liste vide si le fichier n'existe pas encore).

        Chaque ligne est un dictionnaire {nom de colonne: texte}.
        """
        if not os.path.exists(self.chemin):
            return []
        lignes = []
        with open(self.chemin, newline="", encoding="utf-8") as fichier:
            lecteur = csv.reader(fichier)
            en_tete = next(lecteur, [])
            for ligne in lecteur:
                lignes.append(dict(zip(en_tete, ligne)))
        return lignes

    def _verifier_en_tete(self) -> None:
        """Refuse d'écrire dans un fichier dont les colonnes ne correspondent pas."""
        with open(self.chemin, newline="", encoding="utf-8") as fichier:
            en_tete = next(csv.reader(fichier), [])
        if en_tete != self.colonnes:
            raise ErreurProjet(
                f"Le fichier d'historique « {self.chemin} » a d'autres colonnes "
                "(il date d'une version précédente). Supprimez-le puis réessayez."
            )

    def __len__(self) -> int:
        return len(self.lire())

    def __str__(self) -> str:
        return f"Historique de {len(self)} prédiction(s) dans {self.chemin}"
