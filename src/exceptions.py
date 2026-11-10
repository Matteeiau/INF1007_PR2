
class ErreurProjet(Exception):
    """Classe parente de toutes les erreurs propres au projet."""


class ColonneManquanteError(ErreurProjet):
    """Levée quand une colonne déclarée dans config.py est absente du CSV.

    Attributs:
        colonnes (list[str]): noms des colonnes introuvables.
    """

    def __init__(self, colonnes: list[str]) -> None:
        self.colonnes = colonnes
        message = (
            f"Colonnes absentes du fichier CSV : {', '.join(colonnes)}. "
            "Vérifiez les noms dans config.py."
        )
        super().__init__(message)


class DonneeInvalideError(ErreurProjet):
    """Levée quand une valeur est vide, non numérique ou hors intervalle.

    Sert surtout à valider les saisies de l'interface.

    Attributs:
        variable (str): nom de la variable concernée.
        valeur (object): valeur fautive reçue.
    """

    def __init__(self, variable: str, valeur: object, raison: str) -> None:
        self.variable = variable
        self.valeur = valeur
        super().__init__(f"« {variable} » : {raison} (valeur reçue : {valeur!r}).")


class ModeleNonEntraineError(ErreurProjet):
    """Levée quand on demande une prédiction à un modèle pas encore entraîné."""

    def __init__(self) -> None:
        super().__init__(
            "Le modèle n'est pas entraîné : appelez entrainer() avant predire()."
        )