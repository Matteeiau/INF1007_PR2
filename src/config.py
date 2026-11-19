"""Configuration du projet : seul fichier à modifier si les colonnes changent."""

import os

# --- Fichiers et dossiers (relatifs à la racine du projet) ---
CHEMIN_CSV = os.path.join("donnees", "jeux.csv")
DOSSIER_MODELES = "modeles"
DOSSIER_FIGURES = "figures"

# --- Colonnes à plusieurs valeurs « A,B,C » : {colonne du CSV: nouveau nom} ---
# Crée la 1re valeur (ex. plateforme) et le nombre de valeurs (ex. nb_plateformes)
COLONNES_MULTIPLES = {
    "platforms": "plateforme",
    "publisher": "editeur",
    "genres": "genre",
}

# --- Colonnes utilisées par le modèle ---
CIBLE = "metascore"
COLONNES_NUM = ["nb_plateformes", "nb_editeurs"]
COLONNES_CAT = ["genre", "plateforme", "developer", "editeur"]
# Exclues : id et title (identifiants), userscore (connu seulement après la sortie)

# --- Encodage ---
MAX_CATEGORIES = 150  # au-delà, les valeurs rares sont regroupées dans « Autre »

# --- Séparation train / test ---
PROPORTION_TEST = 0.2
GRAINE = 42

# --- Modèle ---
LAMBDA = 0.1  # pénalité Ridge ; 0 = régression linéaire simple