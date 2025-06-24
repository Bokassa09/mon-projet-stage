import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import random
from bobodiff.backward import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, nin, nout):
        self.weight = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin))
        self.bias = Tensor(np.zeros((1, nout)))
    
    def __call__(self, x):
        return x @ self.weight + self.bias
    
    def parameters(self):
        return [self.weight, self.bias]

class MLP(Module):
    def __init__(self, nin, nouts):
        self.layers = []
        dims = [nin] + nouts
        
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1]))
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # ReLU pour toutes les couches sauf la dernière
            if i < len(self.layers) - 1:
                x = x.relu()
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

# Test avec make_moons
def test_bobodiff():
    # Fixer les seeds
    np.random.seed(1337)
    random.seed(1337)
    
    # Créer le dataset
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1  # Convertir en -1, 1
    
    # Visualiser le dataset
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
    plt.title('Dataset original')
    
    # Créer le modèle
    model = MLP(2, [16, 16, 1])
    print(f"Modèle créé avec {len(model.parameters())} paramètres")
    
    # Convertir les données en Tensors
    X_tensor = Tensor(X)
    y_tensor = Tensor(y.reshape(-1, 1))
    
    # Fonction de loss
    def compute_loss():
        # Forward pass
        scores = model(X_tensor)
        
        # SVM hinge loss: max(0, 1 - y * score)
        margins = Tensor(np.ones_like(y.reshape(-1, 1))) + (-y_tensor) * scores
        hinge_losses = margins.relu()
        data_loss = hinge_losses.mean()
        
        # Régularisation L2
        reg_loss = Tensor(0.0)
        for param in model.parameters():
            reg_loss = reg_loss + (param * param).sum()
        reg_loss = reg_loss * 1e-4
        
        total_loss = data_loss + reg_loss
        
        # Accuracy
        predictions = scores.valeur > 0
        accuracy = np.mean((y.reshape(-1, 1) > 0) == predictions)
        
        return total_loss, accuracy
    
    # Boucle d'entraînement
    losses = []
    accuracies = []
    
    for epoch in range(100):
        # Forward + backward
        total_loss, accuracy = compute_loss()
        
        # Zero gradients
        model.zero_grad()
        
        # Backward pass
        total_loss.backward()
        
        # SGD update
        learning_rate = 1.0 - 0.9 * epoch / 100
        for param in model.parameters():
            param.valeur -= learning_rate * param.grad
        
        losses.append(total_loss.valeur.item())
        accuracies.append(accuracy)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.valeur:.4f}, Accuracy = {accuracy:.2%}")
    
    # Visualiser la frontière de décision
    plt.subplot(1, 2, 2)
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Prédictions sur la grille
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = Tensor(mesh_points, requires_grad=False)
    mesh_scores = model(mesh_tensor)
    Z = mesh_scores.valeur > 0
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title('Frontière de décision')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    plt.tight_layout()
    plt.show()
    
    # Graphique des métriques
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Accuracy pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Accuracy finale: {accuracies[-1]:.2%}")

# Lancer le test
if __name__ == "__main__":
    test_bobodiff()