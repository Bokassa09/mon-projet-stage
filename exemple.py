from forward import AutoDiff
from utile import variable, constante
import numpy as np

def main():
    x=variable(2)
    # Exemple 1: f(x) = x^2
    f1 = x * x  # ou x**2

    assert np.isclose(f1.valeur,4)
    assert np.isclose(f1.derive,4)

    print(" le test a reussi !")

    print("-----------------------------------------")

    # Exemple 2
    # teste sur la fonction definie dans le rapport f(x1,x2)=x1^2*x2+cos(x2)
    x1 = variable(1)  
    x2 = constante(2)  
    f = (x1*x1)*x2 + x2.cos()

    valeur_theorique = 1*1*2 + np.cos(2)
    derivee_theorique = 2*1*2 
    assert np.isclose(f.valeur, valeur_theorique )
    assert np.isclose(f.derive, derivee_theorique )
    print("Le test pour la fonction du rapport a reussi 'x1' ")


    print("Valeur théorique:", valeur_theorique)
    print("Dérivée théorique :", derivee_theorique)
    print("\n")
    print("Résultat AutoDiff pour valeur:", f.valeur)
    print("Résultat AutoDiff pour derive",  f.derive)

    x1 = constante(1)  
    x2 = variable(2)  
    f = (x1*x1)*x2 + x2.cos()
    
    
    derive_theorique = 1 - np.sin(2)
    assert np.isclose(f.derive, derive_theorique )
    print("Le test pour la fonction du rapport a reussi 'x2' ")
    print("\n")
    
    print(f"Résultat AutoDiff pour derive= {f.derive}")
    print(f"Dérivée théorique = {derive_theorique}")
    
main()

