import torch
import time

def test_forward_pytorch():
    x1 = torch.tensor(1.0, requires_grad=True)
    x2 = torch.tensor(2.0, requires_grad=True)
    x3 = torch.tensor(3.0, requires_grad=True)

    debut = time.time()
    for i in range(1):
        f = ((x1 * torch.sin(x2) + torch.log(x3 * x3 + 1))**2 + torch.exp(x1 * x3)) * torch.cos(x2)
 
    f.backward()  # Pour calculer les gradients
    fin = time.time()
    print(f"Temps d'exécution (PyTorch) : {fin - debut} secondes")
    print("f =", f.item())
    print("df/dx1 =", x1.grad.item())
    print("df/dx2 =", x2.grad.item())
    print("df/dx3 =", x3.grad.item())
if __name__ == "__main__":
    test_forward_pytorch()