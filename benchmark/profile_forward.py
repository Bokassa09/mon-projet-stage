import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import cProfile
import pstats
from bobodiff.utile import variable, constante


def test_forward():

    x1 = variable(1.0)
    x2 = constante(2.0)
    x3 = constante(3.0)
    
    debut= time.time()
    for i in range(100000):
        f = ((x1 * x2.sin() + (x3 * x3 + 1).log())**2 + (x1 * x3).exp()) * x2.cos()
    fin= time.time()
    print(f"Temps d'exécution : {fin - debut:.4f} secondes")
    print(f"{f}")
    print("\n")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    test_forward()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumtime").print_stats(20)
    profiler.dump_stats("forward.prof")

