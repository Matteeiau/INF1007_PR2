"""Tests de la régression, des métriques et de la sauvegarde."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import modele as mod
from src.encodeur import Encodeur
from src.exceptions import ErreurProjet, ModeleNonEntraineError


@pytest.fixture
def donnees_bruitees() -> tuple[np.ndarray, np.ndarray]:
    """X avec intercept + 3 variables, y linéaire bruité."""
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    y = X @ np.array([70.0, 2.0, -3.0, 0.5]) + rng.normal(0, 1, 200)
    return X, y


# --- Régression -------------------------------------------------------------

def test_retrouve_une_droite_exacte() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    X = np.column_stack([np.ones(4), x])
    m = mod.RegressionLineaire()
    m.entrainer(X, 3 + 2 * x)
    assert m.beta == pytest.approx([3.0, 2.0])


def test_meme_beta_que_sklearn(donnees_bruitees) -> None:
    X, y = donnees_bruitees
    m = mod.RegressionLineaire()
    m.entrainer(X, y)
    sk = LinearRegression(fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(m.beta, sk.coef_, rtol=1e-8)


def test_ridge_reduit_les_coefficients(donnees_bruitees) -> None:
    X, y = donnees_bruitees
    simple, ridge = mod.RegressionLineaire(), mod.RegressionLineaire(lambda_=500)
    simple.entrainer(X, y)
    ridge.entrainer(X, y)
    assert np.linalg.norm(ridge.beta[1:]) < np.linalg.norm(simple.beta[1:])


def test_predire_avant_entrainer() -> None:
    with pytest.raises(ModeleNonEntraineError):
        mod.RegressionLineaire().predire(np.ones((2, 2)))


def test_matrice_singuliere_message_clair() -> None:
    X = np.column_stack([np.ones(5), np.ones(5)])  # deux colonnes identiques
    with pytest.raises(ErreurProjet, match="LAMBDA"):
        mod.RegressionLineaire().entrainer(X, np.arange(5.0))


def test_lambda_negatif() -> None:
    with pytest.raises(ValueError):
        mod.RegressionLineaire(lambda_=-1)


def test_chronometre_mesure_la_duree(donnees_bruitees) -> None:
    m = mod.RegressionLineaire()
    m.entrainer(*donnees_bruitees)
    assert m.duree_entrainement is not None and m.duree_entrainement >= 0


def test_coefficients_nommes(donnees_bruitees) -> None:
    m = mod.RegressionLineaire()
    m.entrainer(*donnees_bruitees)
    coefs = m.coefficients(["intercept", "a", "b", "c"])
    assert coefs["b"] == pytest.approx(-3.0, abs=0.3)
    with pytest.raises(ValueError):
        m.coefficients(["trop", "court"])


# --- Métriques (NumPy vs scikit-learn) --------------------------------------

def test_metriques_identiques_a_sklearn() -> None:
    y = np.array([70.0, 80.0, 65.0, 90.0])
    p = np.array([72.0, 78.0, 70.0, 85.0])
    assert mod.mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert mod.rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)))
    assert mod.r2(y, p) == pytest.approx(r2_score(y, p))


def test_r2_cas_limites() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mod.r2(y, y) == pytest.approx(1.0)
    assert mod.r2(y, mod.reference(y, 3)) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        mod.r2(np.array([5.0, 5.0]), np.array([4.0, 6.0]))


# --- Sauvegarde -------------------------------------------------------------

def test_sauvegarde_aller_retour(tmp_path) -> None:
    train = pd.DataFrame({"annee": [2000.0, 2010.0, 2020.0, 2015.0],
                          "genre": ["RPG", "Action", "RPG", "Sport"],
                          "metascore": [70.0, 75.0, 80.0, 60.0]})
    enc = Encodeur(["annee"], ["genre"])
    enc.apprendre(train)
    X = enc.transformer(train)
    m = mod.RegressionLineaire(lambda_=1.0)
    m.entrainer(X, train["metascore"].to_numpy())

    mod.sauvegarder(m, enc, {"mae_modele": np.float64(2.5)}, str(tmp_path))
    m2, enc2, metriques = mod.charger(str(tmp_path))

    np.testing.assert_allclose(m2.predire(enc2.transformer(train)), m.predire(X))
    assert metriques["mae_modele"] == 2.5


def test_charger_dossier_vide(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.charger(str(tmp_path))
