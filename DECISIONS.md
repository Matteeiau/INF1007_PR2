# Fiche de décisions — Projet 2 INF1007

Équipe : Matteo Karam, Wassef Dahani
Les cases entre crochets `[ ]` sont à compléter (chiffres mesurés sur le vrai jeu de données).

## Volet 1 — Jeu de données et problème

| Choix | Options envisagées | Choix retenu et justification |
|---|---|---|
| **Domaine** | NBA/Voleyball/Jeux | **Jeux vidéo.** Un domaine que nous connaissons, ce qui aide à juger si les prédictions sont plausibles. Usage : un studio ou un joueur estime la note critique probable d'un jeu à partir de ses caractéristiques. |
| **Jeu de données** | Pas d'autres jeux de données envisagé | **Metacritic** entre 500 et 50 000. Contient des variables numériques et catégorielles, sans donnée personnelle. |
| **Cible et type de problème** | Régression sur `metascore` ; classification en paliers (< 50, 50–74, ≥ 75) | **Régression sur `metascore`.** La note est numérique, des paliers perdraient de l'information, et les classes seraient déséquilibrées (peu de jeux sous 50). |
| **Métrique** | RMSE, MAE, R² | **MAE (principale) et R².** La MAE s'explique à un non-informaticien (« on se trompe en moyenne de X points »). Le R² de la référence vaut environ 0, ce qui rend la comparaison directe. |

## Volet 2 — Préparation et modèle

| Choix | Options envisagées | Choix retenu et justification |
|---|---|---|
| **Colonnes exclues** | Garder toutes les colonnes | **`id` et `title` exclus** : ce sont des identifiants. **`userscore` exclu** : il n'est connu qu'après la sortie, donc ce serait une fuite d'information. Mesure : avec, MAE = [x] ; sans, MAE = [y]. Le gain venait de la fuite. |
| **Cellules à plusieurs valeurs** | Garder la combinaison brute ; une colonne par plateforme | **1ʳᵉ valeur et nombre de valeurs** (`plateforme`, `nb_plateformes`, `editeur`, `nb_editeurs`, `genre`). Les combinaisons brutes donnaient trop de catégories uniques. |
| **Variables d'entrée** | Ajouter `nb_genres` | **6 variables : 2 numériques et 4 catégorielles.** [Résultat de l'essai avec `nb_genres`.] **Limite :** `nb_plateformes` reflète en partie le succès d'un jeu (rééditions après la sortie). |
| **Nettoyage** | Imputer les valeurs manquantes | **Suppression** des lignes sans cible ou sans valeur numérique, et des doublons. Catégories vides remplacées par « Unknown ». Imputer la cible fausserait l'évaluation. [N avant] → [N après] lignes. |
| **Séparation et référence** | 70/30 ; 80/20 | **80/20, graine 42.** La référence prédit toujours la moyenne de l'entraînement. |
| **Modèle** | Régression linéaire ; forêt aléatoire ; k plus proches voisins | **Régression linéaire.** Les coefficients s'interprètent directement (« +X points pour tel studio »). |
| **Implémentation** | `LinearRegression` de scikit-learn ; NumPy | **NumPy** (`np.linalg.solve`). Elle couvre le chapitre 12 et l'annexe D. Un test vérifie que les coefficients sont égaux à ceux de scikit-learn. |
| **Encodage** | One-hot complet ; 40, 150 catégories | **One-hot des 150 catégories les plus fréquentes**, les autres regroupées dans « Autre ». La catégorie de référence n'a pas de colonne, pour que $X^TX$ reste inversible. MAE : 40 → [x], 150 → [y]. |
| **Régularisation** | `LAMBDA` = 0 ; 0,1 ; 1 ; 10 ; 100 | **Ridge, `LAMBDA` = 0,1.** Réduit les coefficients des catégories rares. MAE : [valeurs mesurées]. |

## Conception et interface

| Choix | Options envisagées | Choix retenu et justification |
|---|---|---|
| **Structure du code** | Un seul script ; modules séparés | **10 modules.** Les noms de colonnes sont tous dans `config.py`, et `service.py` sépare la logique du bouton « Prédire » de l'affichage, ce qui la rend testable sans fenêtre. |
| **Héritage** | Une seule classe avec des `if` sur le type | **`Variable` (abstraite) → `VariableNumerique` / `VariableCategorielle`.** Chacune valide sa saisie à sa façon, et l'interface les traite sans connaître leur type. |
| **Outil du chapitre 7** | Générateur, `lambda`, décorateur | **Décorateur `@chronometre`** sur `entrainer()`. |
| **Technologie d'interface** | Tkinter, Streamlit, Gradio | **Tkinter.** Inclus avec Python, fonctionne en local sans serveur, et Matplotlib s'y intègre avec `FigureCanvasTkAgg`. |
| **Graphique de l'interface** | Importance des variables ; prédit contre réel ; position dans la distribution | **Histogramme des vraies notes**, avec la prédiction et une bande ± MAE. C'est le graphique le plus parlant pour un non-informaticien : il situe le jeu par rapport aux autres et montre l'incertitude. |
| **Historique** | CSV, JSON | **CSV**, lisible dans un tableur. L'en-tête est vérifié avant l'écriture, pour éviter de mélanger des versions différentes du modèle. |
| **Note prédite hors de [0, 100]** | Laisser la valeur brute | **Ramenée dans [0, 100]**, puisqu'une régression linéaire n'est pas bornée. |
