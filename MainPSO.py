from PSO import ParticleSwarmOptimization as PSO
import numpy as np

if __name__ == "__main__":
     bound = np.tile([[-600], [600]], 25)
     pso = PSO(60, 25, bound, 1000, [0.7298, 1.4962, 1.4962])
     pso.solve()