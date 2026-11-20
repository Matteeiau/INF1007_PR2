# Journal d'utilisation de l'IA — Projet 2 INF1007

Outil principal : **Claude (Anthropic)**, utilisé en conversation.
Chaque réponse a été vérifiée par un test, une exécution ou la documentation officielle avant d'être gardée. Nous restons responsables de chaque ligne remise et pouvons l'expliquer.

| # | Date | Outil | Ce que nous avons demandé | Réponse reçue | Vérification faite | Décision |
|---|---|---|---|---|---|---|
| 1 | 2026-11-06 | Claude | Si notre idée (prédire le metascore à partir des données Metacritic) respectait l'énoncé. | Oui, c'est un problème de régression. L'IA a signalé que le nom du jeu n'est pas une variable, que le `userscore` serait une fuite d'information, et que le développeur a trop de valeurs distinctes. | Relecture des sections 2 et 3 de l'énoncé (critères, référence, évaluation). | **Gardé.** |
| 2 | 2026-11-06 | Claude | Une explication de la régression linéaire et un pseudo-code. | Moindres carrés, équations normales $X^TX\beta = X^Ty$, encodage one-hot, pseudo-code par fonction. | Comparaison avec notre cours d'algèbre linéaire et avec la doc de `numpy.linalg.solve`. | **Gardé.** Nous avons choisi d'implémenter la régression à la main. |
| 3 | 2026-11-10 | Claude | Une relecture de notre première version de `donnees.py`. | Trois erreurs : `import os` manquant, `drop_dupicates` mal écrit, et des fonctions qui renvoyaient `None` au lieu du DataFrame. | Exécution des tests : `test_nettoyer_complet` échouait bien (`AttributeError`, puis `NoneType`). | **Modifié** : notre code a été corrigé. |
| 4 | 2026-11-10 | Claude | Des tests pour `separer` et les graphiques. | Des tests qui appelaient `tracer_cible_par_categorie` sans l'argument `max_categories`. | Exécution : `TypeError`, car notre signature n'avait pas de valeur par défaut. **La réponse de l'IA ne correspondait pas à notre code.** | **Modifié** : argument ajouté dans le test. |
| 5 | 2026-11-10 | Claude | La classe `Encodeur`. | D'abord un squelette et 11 tests écrits à partir de la spécification, puis la classe complète. | Les tests ont été écrits avant le code, comme le recommande l'énoncé. Relecture du cas « catégorie de référence » (colinéarité avec l'intercept). | **Gardé.** |
| 6 | 2026-11-17 | Claude | `main.py`, alors que l'IA ne connaissait pas encore notre `modele.py`. | Un `main.py` avec des noms de fonctions supposés, et des tests. | **Modifié** : commande de lancement corrigée, et notée dans le README. |
| 7 | 2026-11-17 | Claude | `modele.py` (régression, métriques, sauvegarde). | Une régression NumPy avec Ridge, le décorateur `@chronometre`, des métriques vectorisées et 12 tests. | `test_meme_beta_que_sklearn` (coefficients égaux à ceux de `LinearRegression` à $10^{-8}$ près) ; métriques comparées à `sklearn.metrics`. | **Gardé.** |
| 8 | 2026-11-16 | Claude | Raccourcir les docstrings de `encodeur.py`. | Une version avec une docstring d'une ligne par méthode. L'IA a signalé que le chapitre 6 exige reçoit / renvoie / échoue. | Relecture de l'exigence du chapitre 6. | **Modifié** : docstrings complètes rétablies avant la remise. [Vérifier que c'est fait.] |
| 9 | 2026-11-19 | Claude | Une validation de la version finale. | Logique correcte. À corriger : pas de tests pour `variables`, `historique` et `service` ; docstrings trop courtes ; `matplotlib>=3.9` requis. | Exécution de nos fichiers dans un dossier neuf et test manuel des saisies invalides. | **Modifié** : [tests ajoutés, docstrings complétées]. |
| 10 | 2026-11-19 | Claude | L'IA a servi pour `variables.py`, `service.py`, `historique.py` et `interface.py` pour adapté leur code initial au corps du projet et debug | | | |

## Bilan

L'IA nous a fait gagner du temps sur la structure du code et sur les explications mathématiques. Elle s'est aussi trompée : l'entrée 4 montre un test qui ne correspondait pas à notre code, et l'entrée 6 du code écrit avec des noms supposés. Les tests et l'exécution ont repéré ces erreurs tout de suite. Nous avons appris à faire écrire les tests à partir de la spécification avant le code (entrée 5), plutôt que d'après le code.
