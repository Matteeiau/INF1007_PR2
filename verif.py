import pandas as pd
from src import config as cfg
from src.donnees import charger_csv, extraire_variables_multiples, nettoyer
from src.modele import charger

modele, encodeur, _ = charger("modeles")

# Mêmes préparations que pendant l'entraînement
brut = extraire_variables_multiples(charger_csv(cfg.CHEMIN_CSV), cfg.COLONNES_MULTIPLES)
propre = nettoyer(brut, cfg.COLONNES_NUM, cfg.COLONNES_CAT, cfg.CIBLE)

# Chercher un jeu par son titre (nettoyer garde les mêmes index que brut)
idx = brut.index[brut["title"].str.contains("Zelda", case=False)]
lignes = propre.loc[propre.index.intersection(idx)]

prediction = modele.predire(encodeur.transformer(lignes))
for (i, ligne), p in zip(lignes.iterrows(), prediction):
    print(brut.loc[i, "title"], "| réel :", ligne[cfg.CIBLE], "| prédit :", round(p, 1))

