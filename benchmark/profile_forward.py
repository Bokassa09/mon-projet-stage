from memory_profiler import profile
import time
#import cProfile
#import pstats
from bobodiff.utile import variable, constante

@profile
def test_forward():
    x1 = variable(1.0)
    x2 = constante(2.0)
    x3 = constante(3.0)
    f1 = ((x1 * x2.sin() + (x3 * x3 + 1).log())**2 + (x1 * x3).exp()) * x2.cos()

    x1b = constante(1.0)
    x2b = variable(2.0)
    x3b = constante(3.0)
    f2 = ((x1b * x2b.sin() + (x3b * x3b + 1).log())**2 + (x1b * x3b).exp()) * x2b.cos()

    x1c = constante(1.0)
    x2c = constante(2.0)
    x3c = variable(3.0)
    f3 = ((x1c * x2c.sin() + (x3c * x3c + 1).log())**2 + (x1c * x3c).exp()) * x2c.cos()

    debut = time.time()
    for i in range(10000000):
        x1.valeur = 1.0
        i = f1.valeur
        i = f1.derive

        x2b.valeur = 2.0
        i = f2.valeur
        i = f2.derive

        x3c.valeur = 3.0
        i = f3.valeur
        i = f3.derive
    fin = time.time()

    print(f"Temps d'exécution : {fin - debut:.4f} secondes")
    print(f"df/dx1 = {f1.derive:.6f}")
    print(f"df/dx2 = {f2.derive:.6f}")
    print(f"df/dx3 = {f3.derive:.6f}")

if __name__ == "__main__":
    """profiler = cProfile.Profile()
    profiler.enable()
    test_forward()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumtime").print_stats(20)
    profiler.dump_stats("forward.prof")"""
    test_forward()
