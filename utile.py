from forward import AutoDiff

def constante(valeur):

    return AutoDiff(valeur, 0.0)

def variable(valeur):

    return AutoDiff(valeur, 1.0)