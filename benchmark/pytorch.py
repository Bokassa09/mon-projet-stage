from memory_profiler import profile
import torch
import time

@profile
def test_pytorch():
    x1 = torch.tensor(1.0, requires_grad=True)
    x2 = torch.tensor(2.0, requires_grad=True)
    x3 = torch.tensor(3.0, requires_grad=True)
    f = ((x1 * torch.sin(x2) + torch.log(x3 * x3 + 1))**2 + torch.exp(x1 * x3)) * torch.cos(x2)
    debut = time.time()
    for i in range(10000000):
        f.backward(retain_graph=True)
        x1.grad.zero_()
        x2.grad.zero_()
        x3.grad.zero_()
    fin = time.time()
    f.backward()
    print(f"Temps d'exécution (PyTorch, backward uniquement) : {fin - debut} secondes")
    print("f =", f.item())
    print("df/dx1 =", x1.grad.item())
    print("df/dx2 =", x2.grad.item())
    print("df/dx3 =", x3.grad.item())

if __name__ == "__main__":
    test_pytorch()