import os

CHEMIN_CSV = os.path.join("donnees", "games.csv")
CIBLE = "metascore"
COLONNES_NUM = ["annee", "mois"]
COLONNES_CAT = ["plateforme", "genre", "developpeur"]
MAX_CATEGORIES = 40      # au-delà → « Autre »
PROPORTION_TEST = 0.2
GRAINE = 42