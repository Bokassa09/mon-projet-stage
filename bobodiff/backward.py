import numpy as np

class Tensor:
    
    def __init__(self, valeur, requires_grad=True):
        
        self.valeur = np.array(valeur, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.fils = []
        self.op = None
        self._cached_values = {}
        
        
        if self.requires_grad:
            self.grad = np.zeros_like(self.valeur)

    @property
    def shape(self):
        return self.valeur.shape
    
    @property
    def ndim(self):
        return self.valeur.ndim

    def zero_grad(self):
        """Remet les gradients à zéro"""
        if self.requires_grad:
            self.grad = np.zeros_like(self.valeur)
        for child in self.fils:
            if child.requires_grad:
                child.zero_grad()

    def backward(self, gradient=None):
        """Propagation arrière des gradients"""
        if not self.requires_grad:
            return
            
        if gradient is None:
            # Pour le nœud racine, gradient = 1
            if self.valeur.ndim == 0:  # scalaire
                gradient = np.ones_like(self.valeur)
            else:
                gradient = np.ones_like(self.valeur)
        
        # Accumuler le gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.valeur)
        self.grad = self.grad + gradient

        # Propagation selon l'opération
        if self.op == 'add':
            self._backward_add(gradient)
        elif self.op == 'sub':
            self._backward_sub(gradient)
        elif self.op == 'mul':
            self._backward_mul(gradient)
        elif self.op == 'div':
            self._backward_div(gradient)
        elif self.op == 'pow':
            self._backward_pow(gradient)
        elif self.op == 'matmul':
            self._backward_matmul(gradient)
        elif self.op == 'sum':
            self._backward_sum(gradient)
        elif self.op == 'mean':
            self._backward_mean(gradient)
        elif self.op == 'square':
            self._backward_square(gradient)
        elif self.op == 'exp':
            self._backward_exp(gradient)
        elif self.op == 'log':
            self._backward_log(gradient)
        elif self.op == 'sin':
            self._backward_sin(gradient)
        elif self.op == 'cos':
            self._backward_cos(gradient)
        elif self.op == 'sqrt':
            self._backward_sqrt(gradient)
        elif self.op == 'relu':
            self._backward_relu(gradient)
        elif self.op == 'tanh':
            self._backward_tanh(gradient)
        elif self.op == 'sigmoid':
            self._backward_sigmoid(gradient)
        elif self.op == 'neg':
            self._backward_neg(gradient)

    def _backward_add(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad_a = self._sum_to_shape(gradient, self.fils[0].shape)
            self.fils[0].backward(grad_a)
        if len(self.fils) >= 2 and self.fils[1].requires_grad:
            grad_b = self._sum_to_shape(gradient, self.fils[1].shape)
            self.fils[1].backward(grad_b)

    def _backward_sub(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad_a = self._sum_to_shape(gradient, self.fils[0].shape)
            self.fils[0].backward(grad_a)
        if len(self.fils) >= 2 and self.fils[1].requires_grad:
            grad_b = self._sum_to_shape(-gradient, self.fils[1].shape)
            self.fils[1].backward(grad_b)

    def _backward_mul(self, gradient):
        if len(self.fils) >= 2:
            a, b = self.fils[0], self.fils[1]
            if a.requires_grad:
                grad_a = self._sum_to_shape(gradient * b.valeur, a.shape)
                a.backward(grad_a)
            if b.requires_grad:
                grad_b = self._sum_to_shape(gradient * a.valeur, b.shape)
                b.backward(grad_b)

    def _backward_div(self, gradient):
        if len(self.fils) >= 2:
            a, b = self.fils[0], self.fils[1]
            if a.requires_grad:
                grad_a = self._sum_to_shape(gradient / b.valeur, a.shape)
                a.backward(grad_a)
            if b.requires_grad:
                grad_b = self._sum_to_shape((-gradient * a.valeur) / (b.valeur ** 2), b.shape)
                b.backward(grad_b)

    def _backward_pow(self, gradient):
        if len(self.fils) >= 2:
            a, b = self.fils[0], self.fils[1]
            if a.requires_grad:
                grad_a = gradient * b.valeur * (a.valeur ** (b.valeur - 1))
                grad_a = self._sum_to_shape(grad_a, a.shape)
                a.backward(grad_a)
            if b.requires_grad and np.all(a.valeur > 0):
                grad_b = gradient * self.valeur * np.log(a.valeur)
                grad_b = self._sum_to_shape(grad_b, b.shape)
                b.backward(grad_b)

    def _backward_matmul(self, gradient):
        if len(self.fils) >= 2:
            a, b = self.fils[0], self.fils[1]
            if a.requires_grad:
                grad_a = np.matmul(gradient, b.valeur.T)
                a.backward(grad_a)
            if b.requires_grad:
                grad_b = np.matmul(a.valeur.T, gradient)
                b.backward(grad_b)

    def _backward_sum(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            # Étendre le gradient pour correspondre à la forme originale
            grad = np.broadcast_to(gradient, self.fils[0].shape)
            self.fils[0].backward(grad)

    def _backward_mean(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            # Diviser par le nombre d'éléments
            grad = gradient / self.fils[0].valeur.size
            grad = np.broadcast_to(grad, self.fils[0].shape)
            self.fils[0].backward(grad)

    def _backward_square(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad = gradient * 2 * self.fils[0].valeur
            self.fils[0].backward(grad)

    def _backward_exp(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad = gradient * self.valeur
            self.fils[0].backward(grad)

    def _backward_log(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad = gradient / self.fils[0].valeur
            self.fils[0].backward(grad)

    def _backward_sin(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            if 'cos_cache' in self._cached_values:
                cos_val = self._cached_values['cos_cache']
            else:
                cos_val = np.cos(self.fils[0].valeur)
            grad = gradient * cos_val
            self.fils[0].backward(grad)

    def _backward_cos(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            if 'sin_cache' in self._cached_values:
                sin_val = self._cached_values['sin_cache']
            else:
                sin_val = np.sin(self.fils[0].valeur)
            grad = gradient * (-sin_val)
            self.fils[0].backward(grad)

    def _backward_sqrt(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad = gradient / (2 * self.valeur)
            self.fils[0].backward(grad)

    def _backward_relu(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad = gradient * (self.fils[0].valeur > 0).astype(np.float64)
            self.fils[0].backward(grad)

    def _backward_tanh(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad = gradient * (1 - self.valeur ** 2)
            self.fils[0].backward(grad)

    def _backward_sigmoid(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            grad = gradient * self.valeur * (1 - self.valeur)
            self.fils[0].backward(grad)
    
    def _backward_neg(self, gradient):
        if len(self.fils) >= 1 and self.fils[0].requires_grad:
            self.fils[0].backward(-gradient)

    def _sum_to_shape(self, gradient, target_shape):
        """Réduit un gradient à la forme cible (pour le broadcasting)"""
        # Gérer les cas de broadcasting
        ndims_added = gradient.ndim - len(target_shape)
        for i in range(ndims_added):
            gradient = gradient.sum(axis=0)
        
        # Sommer les dimensions qui ont été broadcastées
        for i, (grad_dim, target_dim) in enumerate(zip(gradient.shape, target_shape)):
            if target_dim == 1 and grad_dim > 1:
                gradient = gradient.sum(axis=i, keepdims=True)
        
        return gradient

    # Opérations arithmétiques
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        result = Tensor(self.valeur + other.valeur)
        result.fils = [self, other]
        result.op = 'add'
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        result = Tensor(self.valeur * other.valeur)
        result.fils = [self, other]
        result.op = 'mul'
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        result = Tensor(self.valeur - other.valeur)
        result.fils = [self, other]
        result.op = 'sub'
        return result
    
    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other.__sub__(self)
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        result = Tensor(self.valeur / other.valeur)
        result.fils = [self, other]
        result.op = 'div'
        return result
    
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other.__truediv__(self)
    
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        result = Tensor(self.valeur ** other.valeur)
        result.fils = [self, other]
        result.op = 'pow'
        return result
    
    def __matmul__(self, other):
        """Multiplication matricielle"""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        result = Tensor(np.matmul(self.valeur, other.valeur))
        result.fils = [self, other]
        result.op = 'matmul'
        return result
    
    def __neg__(self):
        """Négation unaire (-x)"""
        result = Tensor(-self.valeur)
        result.fils = [self]
        result.op = 'neg'
        return result

    # Fonctions mathématiques
    def square(self):
        result = Tensor(self.valeur ** 2)
        result.fils = [self]
        result.op = 'square'
        return result
    
    def exp(self):
        result = Tensor(np.exp(self.valeur))
        result.fils = [self]
        result.op = 'exp'
        return result
    
    def log(self):
        result = Tensor(np.log(self.valeur))
        result.fils = [self]
        result.op = 'log'
        return result
    
    def sin(self):
        sin_val = np.sin(self.valeur)
        cos_val = np.cos(self.valeur)
        result = Tensor(sin_val)
        result.fils = [self]
        result.op = 'sin'
        result._cached_values['cos_cache'] = cos_val
        return result
    
    def cos(self):
        cos_val = np.cos(self.valeur)
        sin_val = np.sin(self.valeur)
        result = Tensor(cos_val)
        result.fils = [self]
        result.op = 'cos'
        result._cached_values['sin_cache'] = sin_val
        return result
    
    def sqrt(self):
        result = Tensor(np.sqrt(self.valeur))
        result.fils = [self]
        result.op = 'sqrt'
        return result
    
    def sum(self, axis=None, keepdims=False):
        result = Tensor(np.sum(self.valeur, axis=axis, keepdims=keepdims))
        result.fils = [self]
        result.op = 'sum'
        return result
    
    def mean(self, axis=None, keepdims=False):
        result = Tensor(np.mean(self.valeur, axis=axis, keepdims=keepdims))
        result.fils = [self]
        result.op = 'mean'
        return result

    # Fonctions d'activation pour les réseaux de neurones
    def relu(self):
        result = Tensor(np.maximum(0, self.valeur))
        result.fils = [self]
        result.op = 'relu'
        return result
    
    def tanh(self):
        result = Tensor(np.tanh(self.valeur))
        result.fils = [self]
        result.op = 'tanh'
        return result
    
    def sigmoid(self):
        result = Tensor(1 / (1 + np.exp(-self.valeur)))
        result.fils = [self]
        result.op = 'sigmoid'
        return result

    def __repr__(self):
        return f"Tensor({self.valeur}, grad={self.grad}, shape={self.shape})"