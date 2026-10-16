import rebound
import airball
import numpy as np
from pathlib import Path
from sys import argv

twopi = 2.0 * np.pi

# Make sure to create the data directory if it doesn't exist.
filepath = Path(__file__).parent / "data/secular/"
filepath.mkdir(parents=True, exist_ok=True)


# Function to set up the simulation with Jupiter, Saturn, Uranus, and Neptune.
def setup(ms=[9.55e-04, 2.86e-04, 4.37e-05, 5.15e-05]):
    sim = rebound.Simulation()
    sim.add(m=1)
    sim.add(m=ms[0], a=5.2038, f="uniform")
    sim.add(m=ms[1], a=9.5397, f="uniform")
    sim.add(m=ms[2], a=19.1927, f="uniform")
    sim.add(m=ms[3], a=30.0760, f="uniform")
    sim.move_to_com()
    return sim


Nstars = int(argv[1])
index = int(argv[2])

# Load the previously generated stars from the file.
stars = airball.Stars("data/flyby.stars")

# Gererate the initial conditions for the system and save them.
sim = setup()
filename = str(filepath / f"flyby-{index:05d}-ic.sim")
sim.save_to_file(filename, delete_file=True)

# Simulate the flyby of a star.
res = airball.flyby(sim, stars[index], hash="flybystar")

# Set up the simulation for integrating an additional 20 Myr.
airball.tools.rotate_into_plane(sim, "invariable")
dt = airball.tools.timestep_for_perihelion_resolution(sim)
if not np.isnan(dt):
    sim.dt = dt
sim.integrator = "whckl"
sim.ri_whfast.safe_mode = False
sim.ri_whfast.keep_unsynchronized = True
sim.exit_max_distance = 100
sim.move_to_com()

# Determine the save frequency based on the simulation time step.
step_size = int(np.round((1e4 * twopi) / sim.dt))
filename = str(filepath / f"flyby-{index:05d}.sa")
sim.save_to_file(filename, step=step_size, delete_file=True)
try:
    sim.integrate(sim.t + 2047 * step_size * sim.dt)
except rebound.Escape:
    print("Escape detected, ending simulation.")
    sim.save_to_file(filename, delete_file=False)
