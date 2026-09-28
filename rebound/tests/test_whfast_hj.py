import ctypes
import math
import unittest

import rebound


class TestIntegratorWHFastHJGivenTree(unittest.TestCase):
    def make_sim(self):
        sim = rebound.Simulation()
        sim.G = 1.0
        sim.integrator = "whfast_hj"
        sim.dt = 0.01
        sim.add(m=1.0)
        sim.add(m=0.1, a=1.0, e=0.05)
        sim.add(m=0.0, a=2.0, e=0.1)
        sim.move_to_com()
        return sim

    def hj_tree_string(self, sim):
        clibrebound = rebound.clibrebound
        clibrebound.reb_integrator_whfast_hj_tree_to_string.argtypes = [
            ctypes.POINTER(rebound.Simulation),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        clibrebound.reb_integrator_whfast_hj_tree_to_string.restype = ctypes.c_int

        small = ctypes.create_string_buffer(1)
        required = clibrebound.reb_integrator_whfast_hj_tree_to_string(
            ctypes.byref(sim),
            small,
            ctypes.c_size_t(1),
        )
        self.assertGreaterEqual(required, 0)

        buffer = ctypes.create_string_buffer(required + 1)
        clibrebound.reb_integrator_whfast_hj_tree_to_string(
            ctypes.byref(sim),
            buffer,
            ctypes.c_size_t(required + 1),
        )
        return buffer.value.decode("ascii")

    def test_fixed_tree_is_required(self):
        sim = self.make_sim()
        with self.assertRaises(RuntimeError):
            sim.integrate(0.01, exact_finish_time=0)

    def test_given_tree_is_cached_and_roundtrips(self):
        sim = self.make_sim()
        tree = "[[1,2],3]"
        sim.integrate(0.05, exact_finish_time=0, given_tree=True, tree=tree)
        self.assertEqual(sim.integrator.given_tree, 1)
        self.assertEqual(self.hj_tree_string(sim), tree)

        # Later integration calls reuse the compiled hierarchy.
        sim.integrate(0.1, exact_finish_time=0)
        self.assertEqual(sim.integrator.given_tree, 1)

    def test_given_tree_accepts_nested_pairs(self):
        sim = self.make_sim()
        sim.integrate(0.01, exact_finish_time=0, given_tree=True, tree=[[1, 2], 3])
        self.assertEqual(sim.integrator.given_tree, 1)

    def test_given_tree_accepts_whitespace_and_leading_zeroes(self):
        sim = self.make_sim()
        sim.integrate(0.0, given_tree=True, tree=" \t[ [01, 002] ,\n03 ] ")
        self.assertEqual(self.hj_tree_string(sim), "[[1,2],3]")

    def test_empty_tree_roundtrips(self):
        sim = rebound.Simulation()
        sim.integrator = "whfast_hj"
        setter = rebound.clibrebound.reb_integrator_whfast_hj_set_tree
        setter.argtypes = [ctypes.POINTER(rebound.Simulation), ctypes.c_char_p]
        setter.restype = ctypes.c_int
        self.assertEqual(setter(ctypes.byref(sim), b"[]"), 0)
        self.assertEqual(sim.integrator.given_tree, 1)
        self.assertEqual(self.hj_tree_string(sim), "[]")

    def test_given_tree_accepts_binary_plus_particles_mode(self):
        sim_explicit = self.make_sim()
        sim_mode = self.make_sim()

        sim_explicit.integrate(0.1, exact_finish_time=0, given_tree=True, tree="[[1,2],3]")
        sim_mode.integrate(0.1, exact_finish_time=0, given_tree=True, tree="binary_plus_particles")

        self.assertEqual(sim_mode.integrator.given_tree, 1)
        for particle_explicit, particle_mode in zip(sim_explicit.particles, sim_mode.particles):
            for attr in ("x", "y", "z", "vx", "vy", "vz"):
                self.assertAlmostEqual(
                    getattr(particle_explicit, attr),
                    getattr(particle_mode, attr),
                    delta=1e-14,
                )

    def test_fixed_tree_rejects_mass_changes(self):
        sim = self.make_sim()
        sim.integrate(0.01, exact_finish_time=0, given_tree=True, tree="[[1,2],3]")
        sim.particles[1].m = 0.2
        with self.assertRaises(RuntimeError):
            sim.integrate(0.02, exact_finish_time=0)

    def test_copied_simulation_requires_tree_again(self):
        sim = self.make_sim()
        sim.integrate(0.01, exact_finish_time=0, given_tree=True, tree="[[1,2],3]")
        copied = sim.copy()
        with self.assertRaises(RuntimeError):
            copied.integrate(0.02, exact_finish_time=0)

    def test_valid_replacement_refreshes_fixed_tree(self):
        sim = self.make_sim()
        sim.integrate(0.05, given_tree=True, tree="[[1,2],3]")
        sim.particles[1].m = 0.2
        sim.add(m=0.0, x=4.0, vy=0.5)

        # Compare replacement against a fresh setup with the new count and masses.
        reference = rebound.Simulation()
        reference.integrator = "whfast_hj"
        reference.G = sim.G
        reference.dt = sim.dt
        reference.t = sim.t
        for particle in sim.particles:
            reference.add(particle)

        coordinates = ("x", "y", "z", "vx", "vy", "vz")
        before = [tuple(getattr(p, attr) for attr in coordinates) for p in sim.particles]
        tree = "[3,[[2,1],4]]"
        sim.integrate(sim.t, given_tree=True, tree=tree)
        self.assertEqual(self.hj_tree_string(sim), tree)
        self.assertEqual(before, [tuple(getattr(p, attr) for attr in coordinates) for p in sim.particles])
        sim.integrate(0.1)
        reference.integrate(0.1, given_tree=True, tree=tree)
        self.assertAlmostEqual(sim.t, 0.1)
        for actual, expected in zip(sim.particles, reference.particles):
            for attr in coordinates:
                self.assertAlmostEqual(getattr(actual, attr), getattr(expected, attr), delta=2e-12)

    def test_fixed_tree_rejects_particle_count_changes(self):
        sim = self.make_sim()
        sim.integrate(0.01, given_tree=True, tree="[[1,2],3]")
        sim.add(m=0.0, x=4.0)
        with self.assertRaises(RuntimeError):
            sim.integrate(0.02)


class TestIntegratorWHFastHJPhysics(unittest.TestCase):
    coordinates = ("x", "y", "z", "vx", "vy", "vz")
    # Two massive binaries, two tracers sharing the first binary, one orbiting
    # a single star, and one orbiting the whole system. Zero-mass leaves occur
    # on both sides of a binary node.
    multiple_hosts_tree = [[[6, [[1, 2], 5]], [[7, 3], 4]], 8]

    def make_multiple_hosts(self):
        sim = rebound.Simulation()
        sim.integrator = "whfast_hj"
        sim.dt = 0.002
        sim.add(m=1.0, x=-4.0, y=-0.3, vx=0.3, vy=-0.1)
        sim.add(m=0.5, x=-4.0, y=0.6, vx=-0.6, vy=-0.1)
        sim.add(m=0.8, x=6.0, y=-0.2, vx=0.2, vy=0.15)
        sim.add(m=0.4, x=6.0, y=0.4, vx=-0.4, vy=0.15)
        sim.add(m=0.0, x=-2.0, vy=0.8, z=0.1)
        sim.add(m=0.0, x=-4.0, y=2.0, vx=-0.8, vz=0.05)
        sim.add(m=0.0, x=6.8, y=0.1, vy=1.1)
        sim.add(m=0.0, x=15.0, y=2.0, vy=0.5)
        sim.N_active = 4
        sim.testparticle_type = 0
        return sim

    def assert_particles_close(self, first, second, tolerance=2e-12):
        for attr in self.coordinates:
            self.assertAlmostEqual(getattr(first, attr), getattr(second, attr), delta=tolerance, msg=attr)

    def max_state_error(self, first, second):
        return max(abs(getattr(a, attr) - getattr(b, attr))
                   for a, b in zip(first.particles, second.particles)
                   for attr in self.coordinates)

    def remap_tree(self, tree, indices):
        """Remove absent leaves and renumber retained particles, preserving order."""
        if isinstance(tree, int):
            return indices.get(tree)
        left = self.remap_tree(tree[0], indices)
        right = self.remap_tree(tree[1], indices)
        if left is None:
            return right
        if right is None:
            return left
        return [left, right]

    def test_multiple_hosts_match_isolated_tracers(self):
        initial = self.make_multiple_hosts()
        combined = self.make_multiple_hosts()
        combined.integrate(0.2, given_tree=True, tree=self.multiple_hosts_tree)
        for tracer in range(4, initial.N):
            with self.subTest(tracer=tracer):
                isolated = rebound.Simulation()
                isolated.integrator = "whfast_hj"
                isolated.dt = initial.dt
                selected = [0, 1, 2, 3, tracer]
                for index in selected:
                    isolated.add(initial.particles[index])
                isolated.N_active = 4
                indices = {old + 1: new + 1 for new, old in enumerate(selected)}
                tree = self.remap_tree(self.multiple_hosts_tree, indices)
                isolated.integrate(0.2, given_tree=True, tree=tree)
                for new, old in enumerate(selected):
                    self.assert_particles_close(combined.particles[old], isolated.particles[new])

    def test_interleaved_massless_particles_and_reversed_hosts(self):
        original = self.make_multiple_hosts()
        # With interleaved masses, use all particles in the direct gravity loop.
        original.N_active = -1
        permuted = rebound.Simulation()
        permuted.integrator = "whfast_hj"
        permuted.dt = original.dt
        order = [4, 0, 6, 2, 1, 5, 3, 7]
        for index in order:
            permuted.add(original.particles[index])
        indices = {old + 1: new + 1 for new, old in enumerate(order)}
        tree = self.remap_tree(self.multiple_hosts_tree, indices)
        original.integrate(0.2, given_tree=True, tree=self.multiple_hosts_tree)
        permuted.integrate(0.2, given_tree=True, tree=tree)
        for new, old in enumerate(order):
            self.assert_particles_close(original.particles[old], permuted.particles[new])

    def test_two_body_agrees_with_ias15(self):
        for companion_mass in (0.0, 0.2):
            with self.subTest(companion_mass=companion_mass):
                sim = rebound.Simulation()
                sim.add(m=1.0, x=0.3, z=-0.1, vx=0.2, vy=-0.1)
                sim.add(m=companion_mass, a=1.3, e=0.35, inc=0.4, omega=0.7, f=0.8)
                reference = sim.copy()
                reference.integrator = "ias15"
                sim.integrator = "whfast_hj"
                sim.dt = 0.01
                # A massless primary exercises the opposite relative-coordinate sign.
                tree = [2, 1] if companion_mass == 0.0 else [1, 2]
                sim.integrate(1.0, given_tree=True, tree=tree)
                reference.integrate(1.0)
                self.assertLess(self.max_state_error(sim, reference), 2e-12)

    def test_single_particle_free_drift(self):
        for mass in (0.0, 2.0):
            with self.subTest(mass=mass):
                sim = rebound.Simulation()
                sim.integrator = "whfast_hj"
                sim.dt = 0.01
                sim.add(m=mass, x=1.0, y=2.0, z=-3.0, vx=0.4, vy=-0.2, vz=0.3)
                sim.integrate(0.5, given_tree=True, tree=1)
                p = sim.particles[0]
                for actual, expected in zip((p.x, p.y, p.z, p.vx, p.vy, p.vz),
                                            (1.2, 1.9, -2.85, 0.4, -0.2, 0.3)):
                    self.assertAlmostEqual(actual, expected, delta=2e-14)

    def test_changed_gravitational_constant_is_used(self):
        sim = rebound.Simulation()
        sim.integrator = "whfast_hj"
        sim.dt = 0.01
        sim.add(m=1.0)
        sim.add(m=0.2, a=1.0, e=0.1)
        sim.integrate(0.1, given_tree=True, tree=[1, 2])
        for gravitational_constant in (0.7, 1.3):
            sim.G = gravitational_constant
            reference = sim.copy()
            reference.integrator = "ias15"
            target = sim.t + 0.2
            sim.integrate(target)
            reference.integrate(target)
            self.assertLess(self.max_state_error(sim, reference), 2e-12)

    def test_multiple_hosts_are_time_reversible(self):
        initial = self.make_multiple_hosts()
        sim = self.make_multiple_hosts()
        sim.integrate(0.0, given_tree=True, tree=self.multiple_hosts_tree)
        sim.steps(100)
        sim.dt = -sim.dt
        sim.steps(100)
        self.assertAlmostEqual(sim.t, 0.0, delta=1e-14)
        self.assertLess(self.max_state_error(sim, initial), 2e-12)

    def test_multiple_hosts_converge_to_ias15(self):
        reference = self.make_multiple_hosts()
        reference.integrator = "ias15"
        reference.integrate(0.2)
        errors = []
        for dt in (0.004, 0.002):
            sim = self.make_multiple_hosts()
            sim.dt = dt
            sim.integrate(0.2, given_tree=True, tree=self.multiple_hosts_tree)
            errors.append(self.max_state_error(sim, reference))
        # The symmetric Kepler/kick map has second-order global error.
        self.assertLess(errors[1], 2e-5)
        self.assertGreater(errors[0]/errors[1], 3.8)
        self.assertLess(errors[0]/errors[1], 4.2)

    def test_position_dependent_additional_force_converges(self):
        def additional_force(simulation_pointer):
            simulation = simulation_pointer.contents
            # A harmonic perturbation about the COM has zero net force. This
            # exercises acceleration refresh after the first Kepler half-step.
            com = simulation.com()
            for particle in simulation.particles:
                particle.ax -= 0.1*(particle.x - com.x)
                particle.ay -= 0.1*(particle.y - com.y)
                particle.az -= 0.1*(particle.z - com.z)

        reference = self.make_multiple_hosts()
        reference.integrator = "ias15"
        reference.additional_forces = additional_force
        reference.integrate(0.2)
        errors = []
        for dt in (0.004, 0.002):
            sim = self.make_multiple_hosts()
            sim.dt = dt
            sim.additional_forces = additional_force
            sim.integrate(0.2, given_tree=True, tree=self.multiple_hosts_tree)
            errors.append(self.max_state_error(sim, reference))
        self.assertLess(errors[1], 2e-5)
        self.assertGreater(errors[0]/errors[1], 3.8)
        self.assertLess(errors[0]/errors[1], 4.2)

    def test_hundreds_of_tracers_match_isolated_orbits(self):
        sim = rebound.Simulation()
        sim.integrator = "whfast_hj"
        sim.dt = 0.01
        sim.add(m=1.0)
        sim.add(m=0.1, a=0.5, e=0.1)
        sim.move_to_com()
        host = sim.com()
        for index in range(512):
            sim.add(m=0.0, primary=host, a=2.0 + 0.002*index,
                    e=0.1, inc=0.2, f=2.0*math.pi*index/512)
        sim.N_active = 2
        selected = (2, 257, 513)
        isolated = []
        for tracer in selected:
            reference = rebound.Simulation()
            reference.integrator = "whfast_hj"
            reference.dt = sim.dt
            for index in (0, 1, tracer):
                reference.add(sim.particles[index])
            reference.N_active = 2
            isolated.append(reference)
        sim.integrate(0.1, given_tree=True, tree="binary_plus_particles")
        for tracer, reference in zip(selected, isolated):
            reference.integrate(0.1, given_tree=True, tree=[[1, 2], 3])
            for actual, expected in zip((0, 1, tracer), (0, 1, 2)):
                self.assert_particles_close(sim.particles[actual], reference.particles[expected])


if __name__ == "__main__":
    unittest.main()
