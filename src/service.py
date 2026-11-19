"""Ce que fait le bouton « Prédire », sans aucune fenêtre (donc testable).

C'est ici que l'interface rencontre le modèle : ``Predicteur`` branche
l'``Encodeur`` et la ``RegressionLineaire`` ensemble.
"""

from __future__ import annotations

import pandas as pd

from src import config as cfg
from src.donnees import charger_csv, extraire_variables_multiples, nettoyer
from src.encodeur import Encodeur
from src.historique import Historique
from src.modele import RegressionLineaire
from src.variables import Variable, valider_saisies


class Predicteur:
    """Prédit une note à partir de valeurs déjà validées (contient l'encodeur et le modèle)."""

    def __init__(self, modele: RegressionLineaire, encodeur: Encodeur) -> None:
        """Assemble le modèle et l'encodeur qui l'accompagne."""
        self._modele = modele
        self._encodeur = encodeur

    def predire(self, valeurs: dict) -> float:
        """Renvoie la note prédite pour un jeu décrit par ``valeurs``.

        Args:
            valeurs: une valeur par variable, indexée par nom de colonne.

        Raises:
            ModeleNonEntraineError: si le modèle n'est pas entraîné.
            ErreurProjet: si l'encodeur n'est pas appris.
        """
        ligne = pd.DataFrame([valeurs])
        X = self._encodeur.transformer(ligne)
        return float(self._modele.predire(X)[0])

    def __repr__(self) -> str:
        return f"Predicteur({self._modele!r}, {self._encodeur!r})"


def faire_une_prediction(
    variables: list[Variable],
    predicteur: Predicteur,
    historique: Historique,
    saisies: dict[str, str],
    bornes: tuple[float, float] = (0.0, 100.0),
) -> tuple[dict, float]:
    """Valide les saisies, prédit, puis note la prédiction dans l'historique.

    Args:
        variables: les variables du formulaire.
        predicteur: le modèle prêt à prédire.
        historique: où enregistrer la prédiction.
        saisies: texte tapé, indexé par nom de variable.
        bornes: la note prédite est ramenée dans cet intervalle (une
            régression linéaire peut sortir de 0 à 100).

    Returns:
        Les valeurs validées et la note prédite.

    Raises:
        DonneeInvalideError: si une saisie est vide ou invalide.
        ErreurProjet: si le modèle ou l'historique refuse de travailler.
        OSError: si l'historique ne peut pas être écrit.
    """
    valeurs = valider_saisies(variables, saisies)
    prediction = float(predicteur.predire(valeurs))
    prediction = min(max(prediction, bornes[0]), bornes[1])
    historique.ajouter(valeurs, prediction)
    return valeurs, prediction


def charger_distribution(chemin_csv: str) -> list[float]:
    """Lit les notes réelles du jeu de données, nettoyées comme pour l'entraînement.

    Args:
        chemin_csv: chemin du fichier CSV des jeux.

    Returns:
        La liste des valeurs de la cible (pour l'histogramme de l'interface).

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ColonneManquanteError: s'il manque une colonne attendue.
    """
    df = charger_csv(chemin_csv)
    df = extraire_variables_multiples(df, cfg.COLONNES_MULTIPLES)
    df = nettoyer(df, cfg.COLONNES_NUM, cfg.COLONNES_CAT, cfg.CIBLE)
    return df[cfg.CIBLE].tolist()
