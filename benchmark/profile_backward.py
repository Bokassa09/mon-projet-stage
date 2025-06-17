import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time 
import cProfile
import pstats
from bobodiff.backward import Tensor

def test_backward():

    x1 = Tensor(1.0)
    x2 = Tensor(2.0)
    x3 = Tensor(3.0)

    debut= time.time()
    for i in range(1):
        f = ((x1 * x2.sin() + (x3 * x3 + 1).log())**2 + (x1 * x3).exp()) * x2.cos()
    f.backward()
    fin= time.time()
    print(f"Temps d'exécution : {fin - debut} secondes")
    print(x1.grad)
    print(x2.grad)
    print(x3.grad)
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    test_backward()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumtime").print_stats(20)
    profiler.dump_stats("backward.prof")
    


    
