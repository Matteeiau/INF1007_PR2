"""Tests des variables du formulaire (validation des saisies)."""

import pandas as pd
import pytest

from src.encodeur import Encodeur
from src.exceptions import DonneeInvalideError
from src.variables import (
    Variable,
    VariableCategorielle,
    VariableNumerique,
    construire_variables,
    valider_saisies,
)


@pytest.fixture
def nb_plateformes() -> VariableNumerique:
    return VariableNumerique("nb_plateformes", 1, 7, entier=True, libelle="Nombre de plateformes")


@pytest.fixture
def genre() -> VariableCategorielle:
    return VariableCategorielle("genre", ["RPG", "FPS", "Sport"])


# --- VariableNumerique ------------------------------------------------------

@pytest.mark.parametrize("saisie, attendu", [("3", 3.0), (" 7 ", 7.0), ("1", 1.0), ("2,0", 2.0)])
def test_numerique_valide(nb_plateformes, saisie, attendu) -> None:
    assert nb_plateformes.valider(saisie) == pytest.approx(attendu)


def test_numerique_virgule_decimale() -> None:
    v = VariableNumerique("note", 0, 10)
    assert v.valider("2,5") == pytest.approx(2.5)


@pytest.mark.parametrize("saisie, message", [
    ("", "obligatoire"),
    ("   ", "obligatoire"),
    ("abc", "pas un nombre"),
    ("0", "entre 1 et 7"),
    ("8", "entre 1 et 7"),
    ("nan", "entre 1 et 7"),
    ("inf", "entre 1 et 7"),
    ("2,5", "entier"),
])
def test_numerique_invalide(nb_plateformes, saisie, message) -> None:
    with pytest.raises(DonneeInvalideError, match=message) as info:
        nb_plateformes.valider(saisie)
    assert info.value.variable == "Nombre de plateformes"  # le libellé, lisible par l'utilisateur


def test_numerique_intervalle_inverse() -> None:
    with pytest.raises(ValueError):
        VariableNumerique("x", 10, 1)


# --- VariableCategorielle ---------------------------------------------------

def test_categorielle_valide(genre) -> None:
    assert genre.valider(" FPS ") == "FPS"


@pytest.mark.parametrize("saisie", ["", "fps", "Puzzle"])
def test_categorielle_invalide(genre, saisie) -> None:
    with pytest.raises(DonneeInvalideError):
        genre.valider(saisie)


def test_categorielle_liste_vide() -> None:
    with pytest.raises(ValueError):
        VariableCategorielle("genre", [])


def test_categorielle_copie_des_valeurs() -> None:
    valeurs = ["A", "B"]
    v = VariableCategorielle("x", valeurs)
    valeurs.append("C")
    assert v.valeurs == ("A", "B")


# --- Classe parente, polymorphisme et méthodes spéciales --------------------

def test_variable_est_abstraite() -> None:
    with pytest.raises(TypeError):
        Variable("x")


def test_nom_vide() -> None:
    with pytest.raises(ValueError):
        VariableCategorielle("  ", ["A"])


def test_polymorphisme(nb_plateformes, genre) -> None:
    assert [v.type_champ() for v in (nb_plateformes, genre)] == ["nombre", "liste"]


def test_egalite_et_affichage(nb_plateformes) -> None:
    copie = VariableNumerique("nb_plateformes", 1, 7, entier=True)
    assert copie == nb_plateformes
    assert copie != VariableNumerique("nb_plateformes", 1, 8, entier=True)
    assert copie != VariableCategorielle("nb_plateformes", ["1"])
    assert "entre 1 et 7" in str(nb_plateformes)
    assert repr(nb_plateformes) == "VariableNumerique('nb_plateformes')"


# --- Fonctions du module ----------------------------------------------------

def test_construire_variables_depuis_encodeur() -> None:
    train = pd.DataFrame({
        "nb_plateformes": [1.0, 3.0, 2.0],
        "note_moyenne": [1.5, 2.0, 3.25],
        "genre": ["RPG", "RPG", "FPS"],
    })
    enc = Encodeur(["nb_plateformes", "note_moyenne"], ["genre"])
    enc.apprendre(train)
    variables = construire_variables(enc, {"genre": "Genre"})

    assert [v.nom for v in variables] == ["nb_plateformes", "note_moyenne", "genre"]
    assert variables[0].entier is True          # bornes entières → décompte
    assert variables[1].entier is False         # bornes décimales → nombre réel
    assert variables[2].libelle == "Genre"
    assert set(variables[2].valeurs) == {"RPG", "FPS"}


def test_valider_saisies(nb_plateformes, genre) -> None:
    valeurs = valider_saisies([nb_plateformes, genre], {"nb_plateformes": "2", "genre": "RPG"})
    assert valeurs == {"nb_plateformes": 2.0, "genre": "RPG"}


def test_valider_saisies_champ_absent(nb_plateformes, genre) -> None:
    with pytest.raises(DonneeInvalideError, match="obligatoire"):
        valider_saisies([nb_plateformes, genre], {"nb_plateformes": "2"})
