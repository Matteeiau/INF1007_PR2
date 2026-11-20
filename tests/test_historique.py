"""Tests de l'historique des prédictions (fichier CSV)."""

import pytest

from src.exceptions import ErreurProjet
from src.historique import Historique


@pytest.fixture
def historique(tmp_path) -> Historique:
    return Historique(str(tmp_path / "historique" / "predictions.csv"), ["nb_plateformes", "genre"])


def test_lire_avant_tout_ajout(historique) -> None:
    assert historique.lire() == []
    assert len(historique) == 0


def test_ajouter_cree_dossier_et_en_tete(historique) -> None:
    historique.ajouter({"nb_plateformes": 2.0, "genre": "RPG"}, 73.456)
    lignes = historique.lire()
    assert len(lignes) == 1
    assert lignes[0]["nb_plateformes"] == "2"      # 2.0 écrit comme 2
    assert lignes[0]["genre"] == "RPG"
    assert lignes[0]["prediction"] == "73.5"       # arrondi à 0,1
    assert lignes[0]["date"] != ""


def test_ajouts_successifs_un_seul_en_tete(historique) -> None:
    for note in (60.0, 70.0, 80.0):
        historique.ajouter({"nb_plateformes": 1.0, "genre": "FPS"}, note)
    assert [l["prediction"] for l in historique.lire()] == ["60.0", "70.0", "80.0"]
    assert len(historique) == 3
    assert "3 prédiction(s)" in str(historique)


def test_valeur_decimale_conservee(historique) -> None:
    historique.ajouter({"nb_plateformes": 2.5, "genre": "RPG"}, 50.0)
    assert historique.lire()[0]["nb_plateformes"] == "2.5"


def test_variable_manquante(historique) -> None:
    with pytest.raises(KeyError):
        historique.ajouter({"genre": "RPG"}, 50.0)


def test_fichier_d_une_autre_version(historique) -> None:
    historique.ajouter({"nb_plateformes": 1.0, "genre": "RPG"}, 50.0)
    autre = Historique(historique.chemin, ["annee"])
    with pytest.raises(ErreurProjet, match="autres colonnes"):
        autre.ajouter({"annee": 2010.0}, 70.0)
    assert len(historique) == 1                    # le fichier existant n'a pas été abîmé


def test_fichier_vide_reecrit_l_en_tete(historique) -> None:
    import os
    os.makedirs(os.path.dirname(historique.chemin), exist_ok=True)
    open(historique.chemin, "w", encoding="utf-8").close()
    historique.ajouter({"nb_plateformes": 1.0, "genre": "RPG"}, 50.0)
    assert len(historique.lire()) == 1
