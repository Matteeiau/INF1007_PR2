# Prédiction de la note Metacritic d'un jeu vidéo

Projet 2 — INF1007, Polytechnique Montréal (automne 2026)
Équipe : Matteo Karam, [coéquipier]

## Le projet

L'application estime la note critique (metascore, sur 100) qu'un jeu vidéo recevrait, à partir de ses caractéristiques connues avant sa sortie : genre, plateforme principale, studio de développement, éditeur, nombre de plateformes et nombre d'éditeurs. On décrit le jeu dans un formulaire, et l'application affiche la note estimée, sa marge d'erreur typique et sa position parmi les vrais jeux du jeu de données. Chaque prédiction est enregistrée dans un historique.

Le modèle est une régression linéaire avec pénalité Ridge, écrite à la main avec NumPy en résolvant $(X^TX + \lambda I)\beta = X^Ty$. Un test vérifie que les coefficients obtenus sont égaux à ceux de scikit-learn.

## Jeu de données

- **Source :** [nom du jeu de données], [lien Kaggle]
- **Licence :** [licence], compatible avec un usage éducatif
- **Taille :** [N] jeux après nettoyage. Fichier : `donnees/jeux.csv`.
- **Colonnes retenues :** `genres`, `platforms`, `developer`, `publisher`, et la cible `metascore`.
- **Colonnes exclues :** `id` et `title` (identifiants), et `userscore`, qui n'est connu qu'après la sortie : l'utiliser serait une fuite d'information.

Les colonnes à plusieurs valeurs (par exemple `"PS3,Xbox 360,PC"`) sont réduites à leur première valeur et à leur nombre de valeurs. Les lignes sans note sont retirées, ainsi que les doublons. Le détail des choix se trouve dans [`DECISIONS.md`](DECISIONS.md).

## Installation et lancement

Prérequis : Python 3.10 ou plus récent, avec Tkinter. Sur macOS avec Homebrew : `brew install python-tk`.

```bash
pip install -r requirements.txt
python -m src.main entrainer   # nettoie les données, entraîne, évalue, sauvegarde dans modeles/
python -m src.main             # ouvre l'interface graphique
python -m pytest               # lance les tests
```

Toutes les commandes se lancent depuis la racine du dépôt.

## Résultats

Évaluation sur l'ensemble de test (20 % des données, graine 42) :

| Modèle | MAE (points) | R² |
|---|---|---|
| Référence (moyenne de l'entraînement) | [x] | [x] |
| Régression linéaire Ridge (λ = 0,1) | [x] | [x] |

La MAE se lit simplement : en moyenne, la note prédite s'écarte de [x] points de la vraie note.

Le modèle bat la référence, mais il reste prudent : il sous-estime les très bons jeux et surestime les mauvais. C'est attendu, car la qualité d'un jeu dépend surtout d'informations absentes des données (gameplay, budget, réception). Pour une régression aux moindres carrés, la variance des prédictions vaut $R^2$ fois celle des vraies notes, ce qui explique ce retour vers la moyenne.

## Structure du code

```text
src/
├── config.py       noms de colonnes et réglages : seul fichier à modifier si le CSV change
├── exceptions.py   exceptions du projet (ErreurProjet, DonneeInvalideError, …)
├── donnees.py      lecture, nettoyage, séparation train/test, graphiques d'exploration
├── encodeur.py     Encodeur : DataFrame → matrice X (centrage, one-hot)
├── modele.py       RegressionLineaire, métriques NumPy, sauvegarde joblib + JSON
├── variables.py    Variable → VariableNumerique / VariableCategorielle (validation des saisies)
├── historique.py   Historique des prédictions (CSV)
├── service.py      logique du bouton « Prédire », testable sans fenêtre
├── interface.py    fenêtre Tkinter (affichage uniquement)
└── main.py         point d'entrée
tests/              tests pytest de la logique (hors interface)
modeles/            modèle entraîné (modele.joblib) et métadonnées (metadonnees.json)
figures/            graphiques d'exploration
```

L'interface ne contient aucun calcul : elle appelle `service.py`, qui valide, prédit et écrit l'historique. C'est ce qui permet de tester la logique sans ouvrir de fenêtre.

## Tests et couverture

```bash
python -m coverage run -m pytest
python -m coverage report
```

[N] tests, couverture de [x] % sur `src/` (hors `interface.py`). Une intégration continue GitHub Actions lance les tests à chaque `push`.

## Outil du chapitre 7

Le décorateur `@chronometre` (dans `modele.py`) mesure la durée de l'entraînement et la stocke dans `duree_entrainement`, sans modifier le code de la méthode `entrainer`.

## Documentation consultée

- pandas — [`read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html), [`Series.str.split`](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html)
- scikit-learn — [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- NumPy — [`numpy.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
- Tkinter — [`tkinter.ttk`](https://docs.python.org/3/library/tkinter.ttk.html)
- Matplotlib — [Embedding in Tk](https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html)

L'usage des outils d'IA est décrit dans [`JOURNAL_IA.md`](JOURNAL_IA.md).