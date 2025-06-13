import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bobodiff.forward import AutoDiff
from bobodiff.utile import constante, variable
import numpy as np

# Variable : x1, Constante : x2
x1 = variable(1.0)
x2 = constante(2.0)
f1 = (x1 * x1) * x2 + x2.cos()

print("Cas 1 : dérivée par rapport à x1")
print(f"Résultat AutoDiff - valeur : {f1.valeur}")
print(f"Résultat AutoDiff - dérivée : {f1.derive}")
print()

# Constante : x1, Variable : x2
x1 = constante(1.0)
x2 = variable(2.0)
f2 = (x1 * x1) * x2 + x2.cos()

print("Cas 2 : dérivée par rapport à x2")
print(f"Résultat AutoDiff - valeur : {f2.valeur}")
print(f"Résultat AutoDiff - dérivée : {f2.derive}")
