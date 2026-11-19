"""Variables d'entrée du formulaire : chacune sait valider sa propre saisie.

``Variable`` est la classe parente (abstraite). ``VariableNumerique`` et
``VariableCategorielle`` redéfinissent ``valider`` : c'est le polymorphisme.
Les variables sont construites à partir de l'``Encodeur`` du modèle sauvegardé,
donc le formulaire s'adapte tout seul au modèle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.encodeur import Encodeur
from src.exceptions import DonneeInvalideError


class Variable(ABC):
    """Une variable d'entrée : un nom, un libellé lisible et une règle de validation."""

    def __init__(self, nom: str, libelle: str | None = None) -> None:
        """Crée la variable.

        Args:
            nom: nom de la colonne utilisée par le modèle.
            libelle: nom affiché à l'utilisateur (par défaut, le nom de la colonne).

        Raises:
            ValueError: si le nom est vide.
        """
        if not nom or not nom.strip():
            raise ValueError("Le nom d'une variable ne peut pas être vide.")
        self._nom = nom.strip()
        self._libelle = libelle or self._nom

    @property
    def nom(self) -> str:
        """Nom de la colonne utilisée par le modèle."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Nom affiché à l'utilisateur."""
        return self._libelle

    @abstractmethod
    def valider(self, saisie: str) -> float | str:
        """Vérifie la saisie et renvoie la valeur prête pour le modèle.

        Raises:
            DonneeInvalideError: si la saisie est vide ou invalide.
        """

    @abstractmethod
    def type_champ(self) -> str:
        """Type de champ à afficher : ``"nombre"`` (case à taper) ou ``"liste"``."""

    def _verifier_non_vide(self, saisie: str) -> str:
        """Renvoie la saisie sans espaces autour, ou lève une erreur si elle est vide."""
        texte = saisie.strip()
        if not texte:
            raise DonneeInvalideError(self._libelle, saisie, "ce champ est obligatoire")
        return texte

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._nom!r})"


class VariableNumerique(Variable):
    """Un nombre compris entre un minimum et un maximum (éventuellement entier)."""

    def __init__(self, nom: str, minimum: float, maximum: float,
                 entier: bool = False, libelle: str | None = None) -> None:
        """Crée la variable.

        Args:
            nom: nom de la colonne utilisée par le modèle.
            minimum: plus petite valeur acceptée.
            maximum: plus grande valeur acceptée.
            entier: si vrai, seule une valeur entière est acceptée (un décompte).
            libelle: nom affiché à l'utilisateur.

        Raises:
            ValueError: si le minimum dépasse le maximum.
        """
        super().__init__(nom, libelle)
        if minimum > maximum:
            raise ValueError(f"Intervalle inversé pour {nom} : {minimum} > {maximum}.")
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._entier = entier

    @property
    def intervalle(self) -> tuple[float, float]:
        """Intervalle accepté ``(minimum, maximum)``."""
        return (self._minimum, self._maximum)

    @property
    def entier(self) -> bool:
        """Vrai si seule une valeur entière est acceptée."""
        return self._entier

    def valider(self, saisie: str) -> float:
        """Convertit la saisie en nombre et vérifie intervalle et type entier.

        La virgule décimale (``"2,5"``) est acceptée.

        Raises:
            DonneeInvalideError: saisie vide, non numérique, hors intervalle
                ou non entière alors qu'un entier est exigé.
        """
        texte = self._verifier_non_vide(saisie).replace(",", ".")
        try:
            valeur = float(texte)
        except ValueError:
            raise DonneeInvalideError(self._libelle, saisie, "ce n'est pas un nombre") from None
        # nan et inf passent float() mais échouent ici (toute comparaison avec nan est fausse).
        if not self._minimum <= valeur <= self._maximum:
            raise DonneeInvalideError(
                self._libelle, saisie,
                f"doit être entre {self._minimum:g} et {self._maximum:g}")
        if self._entier and valeur != int(valeur):
            raise DonneeInvalideError(self._libelle, saisie, "doit être un nombre entier")
        return valeur

    def type_champ(self) -> str:
        """Une case où l'on tape un nombre."""
        return "nombre"

    def __eq__(self, autre: object) -> bool:
        if not isinstance(autre, VariableNumerique):
            return NotImplemented
        return (self._nom, self._minimum, self._maximum, self._entier) == (
            autre._nom, autre._minimum, autre._maximum, autre._entier)

    def __str__(self) -> str:
        genre = "nombre entier" if self._entier else "nombre"
        return f"{self._libelle} ({genre} entre {self._minimum:g} et {self._maximum:g})"


class VariableCategorielle(Variable):
    """Un choix parmi une liste de valeurs connues du modèle."""

    def __init__(self, nom: str, valeurs: list[str], libelle: str | None = None) -> None:
        """Crée la variable.

        Args:
            nom: nom de la colonne utilisée par le modèle.
            valeurs: valeurs acceptées (copiées : modifier la liste après coup est sans effet).
            libelle: nom affiché à l'utilisateur.

        Raises:
            ValueError: si la liste de valeurs est vide.
        """
        super().__init__(nom, libelle)
        if not valeurs:
            raise ValueError(f"La liste de valeurs de {nom} ne peut pas être vide.")
        self._valeurs = tuple(valeurs)

    @property
    def valeurs(self) -> tuple[str, ...]:
        """Valeurs acceptées."""
        return self._valeurs

    def valider(self, saisie: str) -> str:
        """Vérifie que la saisie est l'une des valeurs connues (casse respectée).

        Raises:
            DonneeInvalideError: saisie vide ou valeur inconnue.
        """
        texte = self._verifier_non_vide(saisie)
        if texte not in self._valeurs:
            raise DonneeInvalideError(self._libelle, saisie, "choisissez une valeur de la liste")
        return texte

    def type_champ(self) -> str:
        """Une liste déroulante."""
        return "liste"

    def __eq__(self, autre: object) -> bool:
        if not isinstance(autre, VariableCategorielle):
            return NotImplemented
        return (self._nom, self._valeurs) == (autre._nom, autre._valeurs)

    def __str__(self) -> str:
        return f"{self._libelle} (liste de {len(self._valeurs)} choix)"


def construire_variables(encodeur: Encodeur, libelles: dict[str, str] | None = None) -> list[Variable]:
    """Crée les variables du formulaire à partir de l'encodeur du modèle sauvegardé.

    Une variable numérique dont le minimum et le maximum sont entiers est
    considérée comme un décompte : seule une valeur entière est acceptée.

    Args:
        encodeur: encodeur appris (donne les colonnes, intervalles et catégories).
        libelles: noms affichés, par nom de colonne (facultatif).

    Returns:
        Les variables numériques, puis les catégorielles, dans l'ordre de l'encodeur.
    """
    libelles = libelles or {}
    variables: list[Variable] = []
    for nom in encodeur.colonnes_num:
        minimum, maximum = encodeur.intervalle(nom)
        entier = float(minimum).is_integer() and float(maximum).is_integer()
        variables.append(VariableNumerique(nom, minimum, maximum, entier, libelles.get(nom)))
    for nom in encodeur.colonnes_cat:
        variables.append(VariableCategorielle(nom, encodeur.valeurs_possibles(nom), libelles.get(nom)))
    return variables


def valider_saisies(variables: list[Variable], saisies: dict[str, str]) -> dict[str, float | str]:
    """Valide toutes les saisies du formulaire.

    Args:
        variables: les variables du formulaire.
        saisies: texte tapé, indexé par nom de colonne.

    Returns:
        Les valeurs validées, indexées par nom de colonne.

    Raises:
        DonneeInvalideError: dès la première saisie invalide (ou absente).
    """
    valeurs: dict[str, float | str] = {}
    for variable in variables:
        valeurs[variable.nom] = variable.valider(saisies.get(variable.nom, ""))
    return valeurs
