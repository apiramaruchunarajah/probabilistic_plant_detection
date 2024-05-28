from .particle_filter_base import ParticleFilter
from core.resampling.resampler import Resampler

# Modified code from :
# Jos Elfring, Elena Torta, and René van de Molengraft.
# Particle filters: A hands-on tutorial.
# Sensors, 21(2), 2021.


class ParticleFilterSIR(ParticleFilter):
    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm):

        # Initialize particle0 filter base class
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Set SIR specific properties
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()

    def needs_resampling(self):
        """
        Method that determines whether not a core step is needed for the current particle0 filter state estimate.
        The sampling importance core (SIR) scheme resamples every time step hence always return true.

        :return: Boolean indicating whether or not core is needed.
        """
        return True

    def update(self, plants_motion_move_distance, measurement):
        # Loop over all particles
        new_particles = []
        for par in self.particles:
            # Propagate the particle0 state according to the current particle0
            propagated_state = self.propagate_sample(par[1], plants_motion_move_distance)

            # Compute current particle0's weight
            #weight = par[0] * self.compute_likelihood(propagated_state, measurement)

            # Store
            #new_particles.append([weight, propagated_state])
            new_particles.append([par[0], propagated_state])

        # Update particles
        self.particles = self.normalize_weights(new_particles)

        # Resample if needed
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
