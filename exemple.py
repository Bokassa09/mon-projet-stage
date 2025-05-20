from forward import AutoDiff
from utile import variable, constante
import numpy as np

def main():
    x=variable(2)
    # Exemple 1: f(x) = x^2
    f1 = x * x  # ou x**2

    assert np.isclose(f1.valeur,4)
    assert np.isclose(f1.derive,4)

    print(" Tout est bon !")

    print(f"f(x) = x^2")
    print(f"f(2) = {f1.valeur}")
    print(f"f'(2) = {f1.derive}")  
    
main()

