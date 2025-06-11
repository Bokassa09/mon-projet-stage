import numpy as np 

class Tensor:

    def __init__(self,valeur):
        self.valeur=valeur # attribut valeur pour stocker les valeurs des tensor
        self.grad=0.0 
        self.fils=[]
        self.op=None # Opérations *;//;+;-

    def backward(self, gradient=1.0):
        """Docstring for backward
        Fonction générale de rétropropagation
        :param self: Description
        :type self: float
        :param gradient:le gradient qui sera propagé ici c'est egal à 1 car dz/dz=1
        :type gradient: 
        """

        self.grad=self.grad + gradient 

# self.fils[1] c'est le y par exemple z= x + y le parent c'est z et les deux fils c'est x et y avec op="+"
# self.fils[0] c'est le x 

        if self.op=='add':
            self.fils[0].backward(gradient)
            if len(self.fils)>1:
                self.fils[1].backward(gradient)
        
        elif self.op=='sub':
            self.fils[0].backward(gradient) # + gradient pour a
            if len(self.fils)>1:
                self.fils[1].backward(-gradient) # - gradient pour a


        elif self.op=='mul':
            a,b=self.fils
            a.backward(gradient*b.valeur)
            if len(self.fils)>1:
                b.backward(gradient*a.valeur)
        
        elif self.op=='div':
            a,b=self.fils
            a.backward(gradient/b.valeur)
            if len(self.fils)>1:
                b.backward((-gradient*a.valeur)/(b.valeur**2))


        elif self.op=='pow':
            # a^b
            a,b=self.fils
            # derive par rapport a (b*a'*a^b-1)
            if len(self.children) > 1:
                a.backward(gradient*b.valeur*(a.valeur**(b.valeur-1)))
            # derive par rapport b (a^b*ln(a))
                b.backward(gradient*(a.valeur**b.valeur)*np.ln(a.valeur))
            
            else:
                # Si b est une constante
                a.backward(gradient*b.valeur*(a.valeur**(b.valeur-1)))

        elif self.op == 'exp':
            
            self.fils[0].backward(gradient * self.self)
        
        elif self.op == 'log':
            
            self.fils[0].backward(gradient / self.fils[0].valeur)
        
        elif self.op == 'sin':
    
            self.fils[0].backward(gradient * np.cos(self.fils[0].valeur))
        
        elif self.op == 'cos':
            
            self.fils[0].backward(gradient * (-np.sin(self.fils[0].valeur)))
        
        elif self.op == 'sqrt':

            self.fils[0].backward(gradient / (2 * self.valeur))

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.valeur + other.valeur)
        result.fils = [self, other]
        result.op = 'add'
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.valeur * other.valeur)
        result.fils = [self, other]
        result.op = 'mul'
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.valeur - other.valeur)
        result.fils= [self, other]
        result.op = 'sub'
        return result
    
    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__sub__(self)
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.valeur / other.valeur)
        result.fils= [self, other]
        result.op = 'div'
        return result
    
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__truediv__(self)
    
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.valeur ** other.valeur)
        result.fils= [self, other]
        result.op = 'pow'
        return result
    
    def exp(self):
        result = Tensor(np.exp(self.valeur))
        result.fils= [self]
        result.op = 'exp'
        return result
    
    def log(self):
        result = Tensor(np.log(self.valeur))
        result.fils = [self]
        result.op = 'log'
        return result
    
    def sin(self):
        result = Tensor(np.sin(self.valeur))
        result.fils= [self]
        result.op = 'sin'
        return result
    
    def cos(self):
        result = Tensor(np.cos(self.valeur))
        result.fils = [self]
        result.op = 'cos'
        return result
    
    
    def sqrt(self):
        result = Tensor(np.sqrt(self.valeur))
        result.fils = [self]
        result.op = 'sqrt'
        return result
    
    def __repr__(self):
        return f"Tensor({self.valeur}, grad={self.grad})"

        




