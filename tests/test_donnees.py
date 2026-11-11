"""Tests du module donnees."""

import pandas as pd
import pytest

from src import donnees
from src.exceptions import ColonneManquanteError


@pytest.fixture
def df_brut() -> pd.DataFrame:
    """Petit DataFrame qui contient les cas à nettoyer."""
    return pd.DataFrame({
        "nom": ["A", "B", "C", "D"],
        "annee": [2010, 2015, "tbd", 2020],
        "genre": [" RPG ", None, "Action", "RPG"],
        "metascore": [80, 75, 60, None],
    })


def test_charger_csv_fichier_absent() -> None:
    with pytest.raises(FileNotFoundError):
        donnees.charger_csv("fichier_qui_nexiste_pas.csv")


def test_charger_csv_lecture(tmp_path) -> None:
    chemin = tmp_path / "mini.csv"
    chemin.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    assert donnees.charger_csv(str(chemin)).shape == (2, 2)


def test_verifier_colonnes_manquante(df_brut) -> None:
    with pytest.raises(ColonneManquanteError) as info:
        donnees.verifier_colonnes(df_brut, ["annee", "zzz"])
    assert info.value.colonnes == ["zzz"]


def test_nettoyer_complet(df_brut) -> None:
    df = donnees.nettoyer(df_brut, ["annee"], ["genre"], "metascore")
    # "tbd" → NaN → ligne supprimée ; metascore None → ligne supprimée
    assert len(df) == 2
    assert list(df.columns) == ["annee", "genre", "metascore"]
    assert set(df["genre"]) == {"RPG", "Unknown"}  # espaces retirés, None remplacé


@pytest.fixture
def df_100() -> pd.DataFrame:
    """DataFrame propre de 100 lignes."""
    return pd.DataFrame({
        "annee": range(100),
        "genre": ["RPG", "Action"] * 50,
        "metascore": [50 + i % 40 for i in range(100)],
    })


def test_separer_tailles(df_100) -> None:
    train, test = donnees.separer(df_100, 0.2, 42)
    assert len(test) == 20
    assert len(train) + len(test) == len(df_100)


def test_separer_reproductible(df_100) -> None:
    train1, _ = donnees.separer(df_100, 0.2, 42)
    train2, _ = donnees.separer(df_100, 0.2, 42)
    assert train1.index.equals(train2.index)


@pytest.mark.parametrize("proportion", [0, 1, -0.1, 1.5])
def test_separer_proportion_invalide(df_100, proportion) -> None:
    with pytest.raises(ValueError):
        donnees.separer(df_100, proportion, 42)


def test_graphiques_crees(df_100, tmp_path) -> None:
    chemin1 = tmp_path / "figures" / "distribution.png"
    chemin2 = tmp_path / "figures" / "par_genre.png"
    donnees.tracer_distribution_cible(df_100, "metascore", str(chemin1))
    donnees.tracer_cible_par_categorie(df_100, "genre", "metascore", str(chemin2))
    assert chemin1.exists() and chemin2.exists()
