import pytest
from bobodiff.backward import Tensor
import numpy as np

def test_addition_backward():
    """Test de l'addition en mode backward"""
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = x + y
    z.backward()
    assert x.grad == 1.0
    assert y.grad == 1.0

def test_soustraction_backward():
    """Test de la soustraction en mode backward"""
    x = Tensor(5.0)
    y = Tensor(2.0)
    z = x - y
    z.backward()
    assert x.grad == 1.0
    assert y.grad == -1.0

def test_multiplication_backward():
    """Test de la multiplication en mode backward"""
    x = Tensor(3.0)
    y = Tensor(4.0)
    z = x * y
    z.backward()
    assert x.grad == 4.0  
    assert y.grad == 3.0  

def test_division_backward():
    """Test de la division en mode backward"""
    x = Tensor(6.0)
    y = Tensor(2.0)
    z = x / y
    z.backward()
    assert x.grad == 0.5  
    assert np.allclose(y.grad, -1.5)  
def test_pow_backward():
    """Test de la puissance en mode backward"""
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = x**y
    z.backward()
    grad_x = 3.0 * (2.0**2)  
    grad_y = (2.0**3) * np.log(2.0)  
    assert np.allclose(x.grad, grad_x)
    assert np.allclose(y.grad, grad_y)

def test_backward():
    """Test de la puissance avec exposant constant"""
    x = Tensor(2.0)
    z = x**3
    z.backward()
    grad = 3.0 * (2.0**2)  
    assert np.allclose(x.grad, grad)

def test_sin_backward():
    """Test du sinus en mode backward"""
    x = Tensor(np.pi/4)
    z = x.sin()
    z.backward()
    grad = np.cos(np.pi/4) 
    assert np.allclose(x.grad,grad)

def test_cos_backward():
    """Test du cosinus en mode backward"""
    x = Tensor(np.pi/4)
    z = x.cos()
    z.backward()
    grad = -np.sin(np.pi/4) 
    assert np.allclose(x.grad, grad)

def test_log_backward():
    """Test du logarithme naturel en mode backward"""
    x = Tensor(2.0)
    z = x.log()
    z.backward()
    grad = 1.0 / 2.0 
    assert np.allclose(x.grad, grad)

def test_exp_backward():
    """Test de l'exponentielle en mode backward"""
    x = Tensor(1.0)
    z = x.exp()
    z.backward()
    grad = np.exp(1.0) 
    assert np.allclose(x.grad, grad)

def test_sqrt_backward():
    """Test de la racine carrée en mode backward"""
    x = Tensor(4.0)
    z = x.sqrt()
    z.backward()
    grad = 1.0 / (2.0 * np.sqrt(4.0)) 
    assert np.allclose(x.grad, grad)

def test_complete_backward():
    """Test d'une expression complexe combinant plusieurs opérations"""
    x = Tensor(2.0)
    y = Tensor(3.0)
    # z = sin(x^2) + cos(y) * exp(x)
    z = (x**2).sin() + y.cos() * x.exp()
    z.backward()
    
    grad_x = 2.0 * 2.0 * np.cos(4.0) + np.cos(3.0) * np.exp(2.0)
    
    grad_y = -np.sin(3.0) * np.exp(2.0)
    
    assert np.allclose(x.grad, grad_x, rtol=1e-10)
    assert np.allclose(y.grad, grad_y, rtol=1e-10)

def test_chaine_backward():
    """Test de la règle de la chaîne"""
    x = Tensor(1.0)
    
    z = (x**2).sin().exp()
    z.backward()
    
    
    grad = np.exp(np.sin(1.0)) * np.cos(1.0) * 2.0
    assert np.allclose(x.grad, grad, rtol=1e-10)

def test_multiple_backward():
    """Test quand une variable est utilisée plusieurs fois"""
    x = Tensor(2.0)
    
    z = x**2 + x**3
    z.backward()
    
    
    grad = 2.0 * 2.0 + 3.0 * 4.0 
    assert np.allclose(x.grad, grad)