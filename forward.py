import numpy as np

class AutoDiff:
    """
    """
    def __init__(self, valeur, derive=1.0):
        self.valeur=valeur
        self.derive=derive
    def __repr__(self):
        return f"AutoDiff : valeur={self.valeur}, deriveé={self.derive}"
    
    # Definition des operateurs de base en mathemtique

    def __add__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        res=AutoDiff(
            valeur=self.valeur + other.valeur
            derive=self.derive + other.derive
        )
        return res

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        res=AutoDiff(
            valeur=self.valeur + other.valeur
            derive=self.derive + other.derive
        )
        return res

    def __sub__(self, other):
        """ Si l'objet se trouve à droite """
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        return other.__sub__(self)

    def __mul__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        res=AutoDiff(
            valeur=self.valeur + other.valeur
            derive=self.derive*other.valeur + self.valeur*other.derive
        )
        return res

    def __rmul__(self, other):
        return self.__mul__(other=)
    
    
    def __truediv__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        res=AutoDiff(
            valeur=self.valeur/other.valeur
            derive=(self.derive*other.valeur - self.valeur*other.derive)/(other.valeur**2)
        )
    

    def __rtruediv__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        return other.__truediv__(self)
    
    def __pow__(self, power):
        if isinstance(power, AutoDiff):
            res=AutoDiff(
                valeur=self.valeur**power.valeur
                derive=self.valeur**power.valeur*(
                power.derive*np.log(self.valeur)+
                power.valeur*self.derive/self.valeur
                )
            )
        else:
            res=AutoDiff(
            valeur=self.valeur**power
            derive=power*self.valeur**(power-1)*self.derive
            )

    # Defintion des fonction mathématiques courantes

    def exp(self):
        valeur_exp=np.exp(self.valeur)

        return AutoDiff(
            valeur=valeur_exp
            derive=valeur_exp*self.derive
        )
    
    def sin(self):

        return AutoDiff(
            valeur=np.sin(self.valeur)
            derive=np.cos(self.valeur)*self.derive
        )
    
    def cos(self):

        return AutoDiff(
            valeur=np.cos(self.valeur)
            derive=-np.sin(self.valeur)*self.derive
        )
    

    def log(self):

        return AutoDiff(
            valeur=np.log(self.valeur)
            derive=self.derive/self.valeur
        )
    
    