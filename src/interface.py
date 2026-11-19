"""Fenêtre Tkinter : formulaire à gauche, graphique et historique à droite.

Ce fichier ne contient que de l'affichage. La validation, la prédiction et
l'écriture de l'historique sont faites par ``service.faire_une_prediction``,
ce qui permet de les tester sans ouvrir de fenêtre.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src import config as cfg
from src.exceptions import ErreurProjet
from src.historique import Historique
from src.modele import charger as charger_modele
from src.service import Predicteur, charger_distribution, faire_une_prediction
from src.variables import Variable, VariableCategorielle, VariableNumerique, construire_variables

AVERTISSEMENT = "Estimation approximative, à titre pédagogique : ce n'est pas la vraie note."
CHEMIN_HISTORIQUE = os.path.join("historique", "predictions.csv")

# Noms affichés pour les colonnes du modèle (sinon : le nom de la colonne).
LIBELLES = {
    "nb_plateformes": "Nombre de plateformes",
    "nb_editeurs": "Nombre d'éditeurs",
    "genre": "Genre",
    "plateforme": "Plateforme principale",
    "developer": "Développeur",
    "editeur": "Éditeur principal",
    "prediction": "Note prédite",
    "date": "Date",
}

# Petite aide affichée à côté du nom du champ.
AIDES = {
    "nb_plateformes": "sur combien de plateformes le jeu sort-il ?",
    "nb_editeurs": "combien d'éditeurs se partagent le jeu ?",
    "plateforme": "la première plateforme annoncée",
    "developer": "« Autre » = studio peu fréquent",
    "editeur": "« Autre » = éditeur peu fréquent",
}


class Application:
    """Fenêtre principale. Elle *contient* le modèle, les variables et l'historique."""

    def __init__(
        self,
        racine: tk.Tk,
        variables: list[Variable],
        predicteur: Predicteur,
        historique: Historique,
        distribution: list[float],
        metriques: dict,
    ) -> None:
        """Construit la fenêtre.

        Args:
            racine: fenêtre Tk créée par l'appelant.
            variables: variables du formulaire.
            predicteur: modèle utilisé par le bouton « Prédire ».
            historique: fichier CSV des prédictions.
            distribution: notes réelles du jeu de données (pour l'histogramme).
            metriques: métriques du modèle (on y lit l'erreur moyenne ``mae_modele``).
        """
        self._racine = racine
        self._variables = variables
        self._predicteur = predicteur
        self._historique = historique
        self._distribution = list(distribution)
        self._erreur_typique = float(metriques.get("mae_modele", 0.0))
        self._champs: dict[str, tk.StringVar] = {}  # nom de colonne -> texte du champ

        racine.title("Prédiction de la note Metacritic d'un jeu vidéo")
        racine.geometry("1150x720")
        racine.minsize(980, 660)
        racine.columnconfigure(1, weight=1)
        racine.rowconfigure(0, weight=1)

        self._construire_formulaire()
        self._construire_droite()
        self._dessiner_graphique()
        self._recharger_historique()

        racine.bind("<Return>", self._touche_entree)

    # ------------------------------------------------------------------
    # Construction de la fenêtre
    # ------------------------------------------------------------------
    def _construire_formulaire(self) -> None:
        """Colonne de gauche : un champ par variable, le bouton, l'erreur et le résultat."""
        cadre = ttk.LabelFrame(self._racine, text=" Décris le jeu ", padding=12)
        cadre.grid(row=0, column=0, sticky="ns", padx=(12, 6), pady=12)

        for variable in self._variables:
            ttk.Label(cadre, text=variable.libelle, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
            ttk.Label(cadre, text=self._aide(variable), foreground="gray").pack(anchor="w")

            texte = tk.StringVar()
            self._champs[variable.nom] = texte
            if isinstance(variable, VariableCategorielle):
                champ = ttk.Combobox(cadre, textvariable=texte, width=34,
                                     values=sorted(variable.valeurs), state="readonly")
            else:
                champ = ttk.Entry(cadre, textvariable=texte, width=36)
            champ.pack(fill="x", pady=(0, 8))

        ttk.Button(cadre, text="Prédire la note", command=self.predire).pack(fill="x", pady=(6, 4))

        self._erreur = ttk.Label(cadre, text="", foreground="#c00000", wraplength=300)
        self._erreur.pack(anchor="w", pady=2)

        self._resultat = ttk.Label(cadre, text="", font=("TkDefaultFont", 22, "bold"))
        self._resultat.pack(anchor="w", pady=(8, 0))
        self._precision = ttk.Label(cadre, text="", wraplength=300)
        self._precision.pack(anchor="w")
        ttk.Label(cadre, text=AVERTISSEMENT, foreground="gray", wraplength=300).pack(anchor="w", pady=(12, 0))

    def _construire_droite(self) -> None:
        """Colonne de droite : le graphique Matplotlib et le tableau d'historique."""
        cadre = ttk.Frame(self._racine)
        cadre.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        cadre.columnconfigure(0, weight=1)
        cadre.rowconfigure(0, weight=3)
        cadre.rowconfigure(1, weight=2)

        self._figure = Figure(figsize=(6, 3.4), dpi=100)
        self._axes = self._figure.add_subplot(111)
        self._canevas = FigureCanvasTkAgg(self._figure, master=cadre)
        self._canevas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        zone = ttk.LabelFrame(cadre, text=" Historique des prédictions ", padding=6)
        zone.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        zone.columnconfigure(0, weight=1)
        zone.rowconfigure(0, weight=1)

        colonnes = self._historique.colonnes
        self._tableau = ttk.Treeview(zone, columns=colonnes, show="headings", height=6)
        for nom in colonnes:
            self._tableau.heading(nom, text=LIBELLES.get(nom, nom))
            largeur = 70 if nom.startswith("nb_") else 120
            self._tableau.column(nom, width=largeur, minwidth=60, anchor="w", stretch=False)
        vertical = ttk.Scrollbar(zone, orient="vertical", command=self._tableau.yview)
        horizontal = ttk.Scrollbar(zone, orient="horizontal", command=self._tableau.xview)
        self._tableau.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self._tableau.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def predire(self) -> None:
        """Appelée par le bouton « Prédire » : valide, prédit, puis met l'écran à jour."""
        saisies = {nom: texte.get() for nom, texte in self._champs.items()}
        try:
            _, prediction = faire_une_prediction(
                self._variables, self._predicteur, self._historique, saisies)
        except ErreurProjet as erreur:
            self._afficher_erreur(str(erreur))
            return
        except OSError as erreur:
            self._afficher_erreur(f"Impossible d'écrire l'historique : {erreur}")
            return

        self._erreur.config(text="")
        self._resultat.config(text=f"{prediction:.0f} / 100")
        self._precision.config(
            text=f"Note Metacritic estimée, à ± {self._erreur_typique:.0f} points près en moyenne.")
        self._dessiner_graphique(prediction)
        self._recharger_historique()

    def _touche_entree(self, evenement: tk.Event) -> None:
        """Appelée quand on appuie sur Entrée (Tkinter donne l'événement en argument)."""
        self.predire()

    def _afficher_erreur(self, message: str) -> None:
        """Affiche le message d'erreur en rouge et efface l'ancien résultat."""
        self._erreur.config(text=message)
        self._resultat.config(text="")
        self._precision.config(text="")

    def _dessiner_graphique(self, prediction: float | None = None) -> None:
        """Histogramme des notes réelles ; avec une prédiction, on y trace sa position."""
        self._axes.clear()
        self._axes.hist(self._distribution, bins=25, color="#9ecae1", edgecolor="white",
                        label="Jeux du jeu de données")
        if prediction is not None:
            self._axes.axvspan(prediction - self._erreur_typique, prediction + self._erreur_typique,
                               color="#d62728", alpha=0.12,
                               label=f"Marge d'erreur typique (± {self._erreur_typique:.0f})")
            self._axes.axvline(prediction, color="#d62728", linestyle="--", linewidth=2,
                               label=f"Prédiction : {prediction:.0f}")
        self._axes.set_xlim(0, 100)
        self._axes.set_title(f"Où se situe la prédiction parmi les {len(self._distribution)} jeux réels ?")
        self._axes.set_xlabel("Note Metacritic (sur 100)")
        self._axes.set_ylabel("Nombre de jeux")
        self._axes.legend(loc="upper left")
        self._figure.tight_layout()
        self._canevas.draw()

    def _recharger_historique(self) -> None:
        """Relit le fichier CSV et affiche les lignes, la plus récente en premier."""
        self._tableau.delete(*self._tableau.get_children())
        try:
            lignes = self._historique.lire()
        except OSError:
            lignes = []
        for ligne in reversed(lignes):
            valeurs = [ligne.get(nom, "") for nom in self._historique.colonnes]
            self._tableau.insert("", "end", values=valeurs)

    def _aide(self, variable: Variable) -> str:
        """Petit texte gris sous le nom du champ : l'aide du domaine et l'intervalle permis."""
        morceaux = []
        if variable.nom in AIDES:
            morceaux.append(AIDES[variable.nom])
        if isinstance(variable, VariableNumerique):
            minimum, maximum = variable.intervalle
            morceaux.append(f"de {minimum:g} à {maximum:g}")
        return " — ".join(morceaux) or " "


def lancer_interface(dossier_modeles: str) -> None:
    """Charge le modèle sauvegardé, construit le formulaire et ouvre la fenêtre.

    Args:
        dossier_modeles: dossier qui contient ``modele.joblib`` et ``metadonnees.json``.

    Raises:
        FileNotFoundError: si le modèle ou le fichier de données est introuvable.
        ErreurProjet: si les données sont inutilisables.
    """
    modele, encodeur, metriques = charger_modele(dossier_modeles)
    variables = construire_variables(encodeur, LIBELLES)
    predicteur = Predicteur(modele, encodeur)
    historique = Historique(CHEMIN_HISTORIQUE, [v.nom for v in variables])
    distribution = charger_distribution(cfg.CHEMIN_CSV)

    racine = tk.Tk()
    Application(racine, variables, predicteur, historique, distribution, metriques)
    racine.mainloop()
