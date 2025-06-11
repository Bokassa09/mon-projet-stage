from backward import Tensor
# Test sur l'exemple du rapport 
x1 = Tensor(1.0)
x2 = Tensor(2.0)

f = (x1 * x1) * x2 + x2.cos()
f.backward()

print("f =", f)
print("x1.grad =", x1.grad)
print("x2.grad =", x2.grad)
