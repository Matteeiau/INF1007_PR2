"""Tests de la classe Encodeur (servent de cahier des charges)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.encodeur import Encodeur
from src.exceptions import ErreurProjet


@pytest.fixture
def train() -> pd.DataFrame:
    """Jeu d'entraînement minimal : 1 numérique, 1 catégorielle (3 valeurs)."""
    return pd.DataFrame({
        "annee": [2000.0, 2010.0, 2020.0, 2010.0, 2010.0, 2000.0],
        "genre": ["RPG", "RPG", "RPG", "Action", "Action", "Sport"],
    })


@pytest.fixture
def encodeur(train) -> Encodeur:
    enc = Encodeur(["annee"], ["genre"], max_categories=40)
    enc.apprendre(train)
    return enc


# --- Cas d'erreur -----------------------------------------------------------

def test_transformer_avant_apprendre(train) -> None:
    enc = Encodeur(["annee"], ["genre"])
    with pytest.raises(ErreurProjet):
        enc.transformer(train)


# --- Cas usuels -------------------------------------------------------------

def test_forme_de_X(encodeur, train) -> None:
    X = encodeur.transformer(train)
    # intercept + annee + (3 genres − 1 référence) = 4 colonnes
    assert X.shape == (6, 4)


def test_colonne_intercept(encodeur, train) -> None:
    X = encodeur.transformer(train)
    assert np.all(X[:, 0] == 1.0)


def test_numerique_centree(encodeur, train) -> None:
    X = encodeur.transformer(train)
    assert X[:, 1].mean() == pytest.approx(0.0)


def test_reference_est_la_plus_frequente(encodeur) -> None:
    # RPG apparaît 3 fois → référence, donc pas de colonne "genre=RPG"
    assert encodeur.valeurs_possibles("genre")[0] == "RPG"
    assert "genre=RPG" not in encodeur.noms_colonnes()


def test_one_hot_correct(encodeur) -> None:
    ligne = pd.DataFrame({"annee": [2010.0], "genre": ["Action"]})
    X = encodeur.transformer(ligne)
    noms = encodeur.noms_colonnes()
    assert X[0, noms.index("genre=Action")] == 1.0
    assert X[0, noms.index("genre=Sport")] == 0.0


def test_noms_alignes_avec_X(encodeur, train) -> None:
    assert len(encodeur.noms_colonnes()) == encodeur.transformer(train).shape[1]


def test_intervalle(encodeur) -> None:
    assert encodeur.intervalle("annee") == (2000.0, 2020.0)


# --- Cas limites ------------------------------------------------------------

def test_categorie_inconnue_donne_des_zeros(encodeur) -> None:
    ligne = pd.DataFrame({"annee": [2010.0], "genre": ["Puzzle"]})
    X = encodeur.transformer(ligne)
    assert np.all(X[0, 2:] == 0.0)  # traitée comme la référence, sans plantage


def test_trop_de_categories_reference_autre(train) -> None:
    enc = Encodeur(["annee"], ["genre"], max_categories=2)
    enc.apprendre(train)
    # 3 genres > 2 : on garde RPG et Action, "Sport" part dans « Autre »
    assert enc.valeurs_possibles("genre") == ["Autre", "RPG", "Action"]
    assert enc.transformer(train).shape == (6, 1 + 1 + 2)


# --- Sauvegarde -------------------------------------------------------------

def test_aller_retour_json(encodeur, train) -> None:
    texte = json.dumps(encodeur.vers_dict())          # doit être sérialisable
    copie = Encodeur.depuis_dict(json.loads(texte))
    np.testing.assert_allclose(copie.transformer(train), encodeur.transformer(train))
