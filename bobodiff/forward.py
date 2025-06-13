import numpy as np

class AutoDiff:
    """
    Definition de la classe Au

    """
    def __init__(self, valeur, derive=1.0):
        """
        Crée un nombre avec sa valeur et sa dérivée.

        Args:
            valeur (float): Valeur de la fonction au point x (i.e. f(x)).
            derive (float, optional): Valeur de la dérivée au point x (i.e. f'(x)). Défaut à 1.0.
        """
        self.valeur=float(valeur)
        self.derive=float(derive)
    def __repr__(self):
        return f"AutoDiff : valeur={self.valeur}, deriveé={self.derive}"
    
    # Definition des operateurs de base en mathemtique
    def __neg__(self):
        return AutoDiff(-self.valeur, -self.derive)
    
    def __add__(self, other):

        """
        Surcharge de l'opérateur + pour l'addition d'objets AutoDiff.

        L'addition est définie selon :
            f(x) + g(x) => valeur = f.valeur + g.valeur
            (f + g)'(x) => derive = f.derive + g.derive

        Args:
            other (AutoDiff): L'autre objet AutoDiff à additionner.

        Returns:
            AutoDiff: Un nouvel objet AutoDiff représentant la somme.
        """

        # Convertir other en AutoDiff s'il ne l'est pas déjà
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other,0.0) # Constante -> dérivée = 0
        
        # Règle pour l'addition: (f + g)' = f' + g'
        res=AutoDiff(
            valeur=self.valeur + other.valeur,
            derive=self.derive + other.derive)
        return res

    # Pour permettre other + self
    def __radd__(self, other):
        return self.__add__(other)
    
    

    # Même principe que __add__
    def __sub__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other,0.0)
        
        res=AutoDiff(
            valeur=self.valeur - other.valeur,
            derive=self.derive - other.derive
        )
        return res

    def __rsub__(self, other):
        """ Si l'objet se trouve à droite """
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        return other.__sub__(self)

    # Même principe que __add__
    def __mul__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other,0.0)
        
        res=AutoDiff(
            valeur=self.valeur*other.valeur,
            derive=self.derive*other.valeur + self.valeur*other.derive
        )
        return res

    def __rmul__(self, other):
        return self.__mul__(other)
    
    
    # Même principe que __add__
    def __truediv__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other,0.0)
        
        res=AutoDiff(
            valeur=self.valeur/other.valeur,
            derive=(self.derive*other.valeur - self.valeur*other.derive)/(other.valeur**2)
        )
        return res
    

    def __rtruediv__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        return other.__truediv__(self)
    

    # Même principe que __add__
    def __pow__(self, power):
        if isinstance(power, AutoDiff):
            res=AutoDiff(
                valeur=self.valeur**power.valeur,
                derive=self.valeur**power.valeur*(
                power.derive*np.log(self.valeur)+
                power.valeur*self.derive/self.valeur
                )
            )
        else:
            res=AutoDiff(
            valeur=self.valeur**power,
            derive=power*self.valeur**(power-1)*self.derive
            )
        
        return res
    

    

    # Defintion des fonction mathématiques courantes

    # Dans cette implémentation, les fonctions mathématiques courantes ont été définies comme
    #  des méthodes de la classe. Par conséquent, elles s’utilisent avec la notation objet, c’est-à-dire x.sin(),
    #  et non avec la syntaxe habituelle sin(x) des fonctions standards en Python.
    def exp(self):
        valeur_exp=np.exp(self.valeur)

        return AutoDiff(
            valeur=valeur_exp,
            derive=valeur_exp*self.derive
        )
    
    def sin(self):

        return AutoDiff(
            valeur=np.sin(self.valeur),
            derive=np.cos(self.valeur)*self.derive
        )
    
    def cos(self):

        return AutoDiff(
            valeur=np.cos(self.valeur),
            derive=-np.sin(self.valeur)*self.derive
        )
    

    def log(self):

        return AutoDiff(
            valeur=np.log(self.valeur),
            derive=self.derive/self.valeur
        )
    def sqrt(self):
        val = np.sqrt(self.valeur)
        return AutoDiff(
        valeur=val,
        derive=(0.5 / val) * self.derive
    )

    
