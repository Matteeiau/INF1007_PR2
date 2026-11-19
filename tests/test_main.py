"""Tests du point d'entrée (pipeline complet, sans interface)."""

import os

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import main


@pytest.fixture
def df_synthetique() -> pd.DataFrame:
    """300 jeux dont la note dépend vraiment de l'année et du genre."""
    rng = np.random.default_rng(0)
    n = 300
    annee = rng.integers(2000, 2024, n)
    genre = rng.choice(["RPG", "Action", "Sport"], n)
    plateforme = rng.choice(["PC", "PS5", "Switch"], n)
    effet_genre = pd.Series(genre).map({"RPG": 8, "Action": 0, "Sport": -6}).to_numpy()
    note = 70 + 0.3 * (annee - 2010) + effet_genre + rng.normal(0, 3, n)
    return pd.DataFrame({
        "nom": [f"Jeu {i}" for i in range(n)],
        "annee": annee,
        "genre": genre,
        "plateforme": plateforme,
        "metascore": note.round(),
    })


@pytest.fixture
def csv_synthetique(df_synthetique, tmp_path) -> str:
    chemin = tmp_path / "jeux.csv"
    df_synthetique.to_csv(chemin, index=False)
    return str(chemin)


@pytest.fixture(autouse=True)
def config_de_test(monkeypatch) -> None:
    """Fixe des colonnes connues, indépendamment de votre config.py."""
    monkeypatch.setattr(cfg, "COLONNES_NUM", ["annee"])
    monkeypatch.setattr(cfg, "COLONNES_CAT", ["genre", "plateforme"])
    monkeypatch.setattr(cfg, "CIBLE", "metascore")


def test_preparer_donnees_tailles(csv_synthetique) -> None:
    df, train, test = main.preparer_donnees(csv_synthetique)
    assert len(train) + len(test) == len(df)
    assert "nom" not in df.columns


def test_modele_bat_la_reference(csv_synthetique) -> None:
    _, train, test = main.preparer_donnees(csv_synthetique)
    _, _, m = main.entrainer_et_evaluer(train, test)
    assert m["mae_modele"] < m["mae_reference"]
    assert m["r2_modele"] > 0.5
    assert m["r2_reference"] == pytest.approx(0.0, abs=0.1)


def test_formater_resultats_contient_les_scores() -> None:
    m = {"n_train": 80, "n_test": 20, "mae_modele": 3.14159,
         "r2_modele": 0.5, "mae_reference": 6.0, "r2_reference": 0.0}
    texte = main.formater_resultats(m)
    assert "3.14" in texte and "Référence" in texte


def test_commande_entrainer_cree_les_fichiers(csv_synthetique, tmp_path) -> None:
    modeles, figures = tmp_path / "modeles", tmp_path / "figures"
    code = main.main(["entrainer", "--csv", csv_synthetique,
                      "--modeles", str(modeles), "--figures", str(figures)])
    assert code == 0
    assert (modeles / "modele.joblib").exists()
    assert (modeles / "metadonnees.json").exists()
    assert (figures / "distribution.png").exists()


def test_csv_absent_message_sans_traceback(tmp_path, capsys) -> None:
    code = main.main(["entrainer", "--csv", str(tmp_path / "absent.csv")])
    assert code == 1
    assert "introuvable" in capsys.readouterr().err


def test_colonne_manquante_message(df_synthetique, tmp_path, capsys) -> None:
    chemin = tmp_path / "incomplet.csv"
    df_synthetique.drop(columns=["genre"]).to_csv(chemin, index=False)
    code = main.main(["entrainer", "--csv", str(chemin),
                      "--modeles", str(tmp_path / "m"), "--figures", str(tmp_path / "f")])
    assert code == 1
    assert "genre" in capsys.readouterr().err


def test_interface_sans_modele(tmp_path, capsys) -> None:
    code = main.main(["interface", "--modeles", str(tmp_path / "vide")])
    assert code == 1
    assert "entrainer" in capsys.readouterr().err
