import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bobodiff.forward import AutoDiff

import pytest
from bobodiff.forward import AutoDiff
from bobodiff.utile import variable, constante
import numpy as np

def test_addition():
    """Test de l'addition en mode forward"""
    x = AutoDiff(2.0, 1.0)
    y = AutoDiff(3.0, 0.0)
    z = x + y
    assert z.valeur == 5.0
    assert z.derive == 1.0

def test_soustraction():
    """Test de la soustraction en mode forward"""
    x = AutoDiff(5.0, 1.0)
    y = AutoDiff(2.0, 0.0)
    z = x - y
    assert z.valeur == 3.0
    assert z.derive == 1.0

def test_multiplication():
    """Test de la multiplication en mode forward"""
    x = AutoDiff(3.0, 1.0)
    y = AutoDiff(4.0, 0.0)
    z = x * y
    assert z.valeur == 12.0
    assert z.derive == 4.0  

def test_division():
    """Test de la division en mode forward"""
    x = AutoDiff(6.0, 1.0)
    y = AutoDiff(2.0, 0.0)
    z = x / y
    assert z.valeur == 3.0
    assert z.derive == 0.5  

def test_pow():
    """Test de la puissance en mode forward"""
    x = AutoDiff(2.0, 1.0)
    z = x**3
    assert z.valeur == 8.0
    assert z.derive == 12.0  

def test_pow():
    """Test de la puissance avec exposant variable"""
    x = AutoDiff(2.0, 1.0)
    y = AutoDiff(3.0, 0.0)
    z = x**y
    assert z.valeur == 8.0
    assert np.allclose(z.derive, 12.0)

def test_sin():
    """Test du sinus en mode forward"""
    x = AutoDiff(np.pi/2, 1.0)
    z = x.sin()
    assert np.allclose(z.valeur, 1.0)
    assert np.allclose(z.derive, 0.0)  

def test_sin2():
    """Test du sinus avec une autre valeur"""
    x = AutoDiff(np.pi/4, 1.0)
    z = x.sin()
    assert np.allclose(z.valeur, np.sqrt(2)/2)
    assert np.allclose(z.derive, np.sqrt(2)/2)  

def test_cos():
    """Test du cosinus en mode forward"""
    x = AutoDiff(np.pi/4, 1.0)
    z = x.cos()
    assert np.allclose(z.valeur, np.sqrt(2)/2)
    assert np.allclose(z.derive, -np.sqrt(2)/2)

def test_cos2():
    """Test du cosinus en zéro"""
    x = AutoDiff(0.0, 1.0)
    z = x.cos()
    assert np.allclose(z.valeur, 1.0)
    assert np.allclose(z.derive, 0.0) 


def test_log():
    """Test du logarithme naturel en mode forward"""
    x = AutoDiff(np.e, 1.0)
    z = x.log()
    assert np.allclose(z.valeur, 1.0)
    assert np.allclose(z.derive, 1.0/np.e) 

def test_exp():
    """Test de l'exponentielle en mode forward"""
    x = AutoDiff(1.0, 1.0)
    z = x.exp()
    assert np.allclose(z.valeur, np.e)
    assert np.allclose(z.derive, np.e) 

def test_exp2():
    """Test de l'exponentielle en zéro"""
    x = AutoDiff(0.0, 1.0)
    z = x.exp()
    assert np.allclose(z.valeur, 1.0)
    assert np.allclose(z.derive, 1.0)  
def test_sqrt():
    """Test de la racine carrée en mode forward"""
    x = AutoDiff(4.0, 1.0)
    z = x.sqrt()
    assert np.allclose(z.valeur, 2.0)
    assert np.allclose(z.derive, 0.25) 

def test_sqrt2():
    """Test de la racine carrée avec une autre valeur"""
    x = AutoDiff(9.0, 1.0)
    z = x.sqrt()
    assert np.allclose(z.valeur, 3.0)
    assert np.allclose(z.derive, 1.0/6.0) 

def test_complexe():
    """Test d'une expression complexe combinant plusieurs opérations"""
    x = AutoDiff(2.0, 1.0)
    y = AutoDiff(3.0, 0.0)
    z = (x**2).sin() + y.cos() * x.exp()
    
    val= np.sin(4.0) + np.cos(3.0) * np.exp(2.0)
    derive = 2.0 * 2.0 * np.cos(4.0) + np.cos(3.0) * np.exp(2.0)
    
    assert np.allclose(z.valeur, val)
    assert np.allclose(z.derive, derive)

def test_chaine():
    """Test de la règle de la chaîne"""
    x = AutoDiff(1.0, 1.0)
    z = (x**2).sin().exp()
    
    val = np.exp(np.sin(1.0))
    derive = np.exp(np.sin(1.0)) * np.cos(1.0) * 2.0
    
    assert np.allclose(z.valeur, val)
    assert np.allclose(z.derive, derive)

def test_multiple():
    """Test quand une variable est utilisée plusieurs fois"""
    x = AutoDiff(2.0, 1.0)
    z = x**2 + x**3
    
    val = 4.0 + 8.0 
    derive = 2.0 * 2.0 + 3.0 * 4.0 
    
    assert np.allclose(z.valeur, val)
    assert np.allclose(z.derive, derive)

def test_operations():
    """Test d'opérations imbriquées"""
    x = AutoDiff(1.0, 1.0)
    # z = log(sqrt(exp(x)))
    z = x.exp().sqrt().log()
    
    val = 0.5
    derive = 0.5
    
    assert np.allclose(z.valeur, val)
    assert np.allclose(z.derive, derive)

def test_zero():
    """Test avec dérivée nulle (constante)"""
    x = AutoDiff(5.0, 0.0)
    z = x.sin() + x.cos() * x.exp()
    
    val = np.sin(5.0) + np.cos(5.0) * np.exp(5.0)
    derive = 0.0  
    
    assert np.allclose(z.valeur, val)
    assert np.allclose(z.derive, derive)
