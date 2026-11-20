"""Tests de la logique du bouton « Prédire » (sans fenêtre)."""

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.encodeur import Encodeur
from src.exceptions import DonneeInvalideError, ModeleNonEntraineError
from src.historique import Historique
from src.modele import RegressionLineaire
from src.service import Predicteur, charger_distribution, faire_une_prediction
from src.variables import construire_variables


@pytest.fixture
def train() -> pd.DataFrame:
    """Note = 60 + 5 × nb_plateformes + 10 si RPG (relation exacte)."""
    nb = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    genre = ["RPG", "RPG", "RPG", "FPS", "FPS", "FPS"]
    note = 60 + 5 * nb + np.where(np.array(genre) == "RPG", 10, 0)
    return pd.DataFrame({"nb_plateformes": nb, "genre": genre, "metascore": note})


@pytest.fixture
def encodeur(train) -> Encodeur:
    enc = Encodeur(["nb_plateformes"], ["genre"])
    enc.apprendre(train)
    return enc


@pytest.fixture
def predicteur(train, encodeur) -> Predicteur:
    modele = RegressionLineaire()
    modele.entrainer(encodeur.transformer(train), train["metascore"].to_numpy())
    return Predicteur(modele, encodeur)


@pytest.fixture
def variables(encodeur):
    return construire_variables(encodeur)


@pytest.fixture
def historique(tmp_path, variables) -> Historique:
    return Historique(str(tmp_path / "h.csv"), [v.nom for v in variables])


# --- Predicteur -------------------------------------------------------------

def test_predicteur_retrouve_la_relation(predicteur) -> None:
    # 60 + 5×2 + 10 = 80
    assert predicteur.predire({"nb_plateformes": 2.0, "genre": "RPG"}) == pytest.approx(80.0)
    assert predicteur.predire({"nb_plateformes": 3.0, "genre": "FPS"}) == pytest.approx(75.0)


def test_predicteur_modele_non_entraine(encodeur) -> None:
    with pytest.raises(ModeleNonEntraineError):
        Predicteur(RegressionLineaire(), encodeur).predire({"nb_plateformes": 1.0, "genre": "RPG"})


# --- faire_une_prediction ---------------------------------------------------

def test_prediction_normale_et_historique(variables, predicteur, historique) -> None:
    valeurs, note = faire_une_prediction(
        variables, predicteur, historique, {"nb_plateformes": "2", "genre": "RPG"})
    assert valeurs == {"nb_plateformes": 2.0, "genre": "RPG"}
    assert note == pytest.approx(80.0)
    assert historique.lire()[0]["prediction"] == "80.0"


def test_saisie_invalide_rien_dans_l_historique(variables, predicteur, historique) -> None:
    with pytest.raises(DonneeInvalideError):
        faire_une_prediction(variables, predicteur, historique, {"nb_plateformes": "abc", "genre": "RPG"})
    assert historique.lire() == []


@pytest.mark.parametrize("bornes, attendu", [((0.0, 70.0), 70.0), ((90.0, 100.0), 90.0)])
def test_note_ramenee_dans_les_bornes(variables, predicteur, historique, bornes, attendu) -> None:
    _, note = faire_une_prediction(
        variables, predicteur, historique, {"nb_plateformes": "2", "genre": "RPG"}, bornes)
    assert note == pytest.approx(attendu)


# --- charger_distribution ---------------------------------------------------

def test_charger_distribution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cfg, "COLONNES_MULTIPLES", {"platforms": "plateforme"})
    monkeypatch.setattr(cfg, "COLONNES_NUM", ["nb_plateformes"])
    monkeypatch.setattr(cfg, "COLONNES_CAT", ["plateforme"])
    monkeypatch.setattr(cfg, "CIBLE", "metascore")
    chemin = tmp_path / "jeux.csv"
    chemin.write_text('platforms,metascore\n"PC,PS5",80\nWii,tbd\nPC,60\n', encoding="utf-8")
    assert sorted(charger_distribution(str(chemin))) == [60.0, 80.0]   # "tbd" retiré


def test_charger_distribution_fichier_absent(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        charger_distribution(str(tmp_path / "absent.csv"))
