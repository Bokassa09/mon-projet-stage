import math 

class Tensor:

    def __init__(self, valeur):
        self.valeur = float(valeur)
        self.grad = 0.0 
        self.fils = []
        self.op = None
        self._cached_values = {}

    def zero_grad(self):
        self.grad = 0.0
        for child in self.fils:
            child.zero_grad()

    def backward(self, gradient=1.0):
        self.grad = self.grad + gradient 

        if self.op == 'add':
            self.fils[0].backward(gradient)
            if len(self.fils) > 1:
                self.fils[1].backward(gradient)
        
        elif self.op == 'sub':
            self.fils[0].backward(gradient)
            if len(self.fils) > 1:
                self.fils[1].backward(-gradient)

        elif self.op == 'mul':
            a, b = self.fils
            a.backward(gradient * b.valeur)
            if len(self.fils) > 1:
                b.backward(gradient * a.valeur)
        
        elif self.op == 'div':
            a, b = self.fils
            b_val= b.valeur * b.valeur
            a.backward(gradient / b.valeur)
            if len(self.fils) > 1:
                b.backward((-gradient * a.valeur) / b_val)

        elif self.op == 'pow':
            a, b = self.fils
            if len(self.fils) > 1:
                val_power = a.valeur ** (b.valeur - 1)
                a.backward(gradient * b.valeur * val_power)
                if a.valeur > 0:
                    b.backward(gradient * self.valeur * math.log(a.valeur))
            else:
                val_power = a.valeur ** (b.valeur - 1)
                a.backward(gradient * b.valeur * val_power)

        elif self.op == 'square':
            self.fils[0].backward(gradient * 2 * self.fils[0].valeur)
            
        elif self.op == 'exp':
            self.fils[0].backward(gradient * self.valeur)
        
        elif self.op == 'log':
            self.fils[0].backward(gradient / self.fils[0].valeur)
        
        elif self.op == 'sin':
            if 'cos_cache' in self._cached_values:
                cos_val = self._cached_values['cos_cache']
            else:
                cos_val = math.cos(self.fils[0].valeur)
            self.fils[0].backward(gradient * cos_val)
        
        elif self.op == 'cos':
            if 'sin_cache' in self._cached_values:
                sin_val = self._cached_values['sin_cache']
            else:
                sin_val = math.sin(self.fils[0].valeur)
            self.fils[0].backward(gradient * (-sin_val))
        
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
        result.fils = [self, other]
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
        result.fils = [self, other]
        result.op = 'div'
        return result
    
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__truediv__(self)
    
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            if other == 0:
                return Tensor(1.0)
            elif other == 1:
                return Tensor(self.valeur)
            elif other == 2:
                return self.square()
            other = Tensor(other)
        
        result = Tensor(self.valeur ** other.valeur)
        result.fils = [self, other]
        result.op = 'pow'
        return result
    
    def square(self):
        result = Tensor(self.valeur * self.valeur)
        result.fils = [self]
        result.op = 'square'
        return result
    
    def exp(self):
        exp_val = math.exp(self.valeur)
        result = Tensor(exp_val)
        result.fils = [self]
        result.op = 'exp'
        return result
    
    def log(self):
        result = Tensor(math.log(self.valeur))
        result.fils = [self]
        result.op = 'log'
        return result
    
    def sin(self):
        sin_val = math.sin(self.valeur)
        cos_val = math.cos(self.valeur)
        result = Tensor(sin_val)
        result.fils = [self]
        result.op = 'sin'
        result._cached_values['cos_cache'] = cos_val
        return result
    
    def cos(self):
        cos_val = math.cos(self.valeur)
        sin_val = math.sin(self.valeur)
        result = Tensor(cos_val)
        result.fils = [self]
        result.op = 'cos'
        result._cached_values['sin_cache'] = sin_val
        return result
    
    def sqrt(self):
        sqrt_val = math.sqrt(self.valeur)
        result = Tensor(sqrt_val)
        result.fils = [self]
        result.op = 'sqrt'
        return result
    
    def __repr__(self):
        return f"Tensor({self.valeur}, grad={self.grad})"

        




