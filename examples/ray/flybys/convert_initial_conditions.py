import pickle
import rebound
import numpy as np

assert rebound.__version__ == "4.4.8"

with open("data/initial_conditions.sims", "rb") as f:
    sims = pickle.load(f)

columns = ["m", "x", "y", "z", "vx", "vy", "vz", "r"]

particles = np.array([
    [
        [getattr(p, name) for name in columns]
        for p in sim.particles
    ]
    for sim in sims
], dtype=np.float64)

np.savez_compressed(
    "data/initial_conditions.npz",
    particles=particles,
    columns=np.array(columns),
    t=np.array([sim.t for sim in sims]),
    G=np.array([sim.G for sim in sims]),
    dt=np.array([sim.dt for sim in sims]),
    N_active=np.array([sim.N_active for sim in sims]),
    source_rebound_version=rebound.__version__,
)