from .forward import AutoDiff

def constante(valeur):
    """
    Crée une variable (valeur avec dérivée=1).
    Une variable est un nombre dont la dérivée par rapport à lui-même est 1

    """

    return AutoDiff(valeur, 0.0)

def variable(valeur):
    """
    Crée une constante (valeur avec dérivée=0).
    Une constante est un nombre dont la dérivée est toujours 0.
    """
    
    return AutoDiff(valeur, 1.0)