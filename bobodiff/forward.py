import math

class AutoDiff:
    """
    Definition de la classe AutoDiff

    """
    # __slots__ = ('valeur', 'derive')

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
    
    
    def __truediv__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other,0.0)
        
        other_val= other.valeur * other.valeur
        
        res=AutoDiff(
            valeur=self.valeur/other.valeur,
            derive=(self.derive*other.valeur - self.valeur*other.derive)/other_val
        )
        return res
    

    def __rtruediv__(self, other):
        if not isinstance(other, AutoDiff):
            other=AutoDiff(other)
        
        return other.__truediv__(self)
    

    def __pow__(self, power):
        if isinstance(power, AutoDiff):

            base_val = self.valeur
            power_val = power.valeur
            
            result_val = base_val ** power_val
            
            
            if base_val > 0:  
                log_base = math.log(base_val)
                derive_val = result_val * (
                    power.derive * log_base + 
                    power_val * self.derive / base_val
                )
            else:
                
                derive_val = 0.0
            
            res = AutoDiff(valeur=result_val, derive=derive_val)
        else:
            # Cas f(x)^c où c est une constante
            if power == 0:
                # Optimisation: x^0 = 1, dérivée = 0
                return AutoDiff(1.0, 0.0)
            elif power == 1:
                # Optimisation: x^1 = x, retourner une copie
                return AutoDiff(self.valeur, self.derive)
            elif power == 2:
                # Optimisation spéciale pour le carré 
                return AutoDiff(
                    valeur=self.valeur * self.valeur,
                    derive=2 * self.valeur * self.derive
                )
            else:
                # Cas général avec optimisation
                base_power_minus_1 = self.valeur ** (power - 1)
                res = AutoDiff(
                    valeur=base_power_minus_1 * self.valeur,  # Évite un calcul de puissance
                    derive=power * base_power_minus_1 * self.derive
                )
        
        return res
    

    def exp(self):
        valeur_exp = math.exp(self.valeur)
        return AutoDiff(
            valeur=valeur_exp,
            derive=valeur_exp * self.derive
        )
    
    def sin(self):
        sin_val = math.sin(self.valeur)
        cos_val = math.cos(self.valeur)
        return AutoDiff(
            valeur=sin_val,
            derive=cos_val * self.derive
        )
    
    def cos(self):
        cos_val = math.cos(self.valeur)
        sin_val = math.sin(self.valeur)
        return AutoDiff(
            valeur=cos_val,
            derive=-sin_val * self.derive
        )
    
    def log(self):
        return AutoDiff(
            valeur=math.log(self.valeur),
            derive=self.derive / self.valeur
        )
    
    def sqrt(self):
        sqrt_val = math.sqrt(self.valeur)
        return AutoDiff(
            valeur=sqrt_val,
            derive=self.derive / (2 * sqrt_val)
        )
    
   