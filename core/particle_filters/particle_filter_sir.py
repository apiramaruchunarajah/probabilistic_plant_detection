from .particle_filter_base import ParticleFilter
from core.resampling.resampler import Resampler


class ParticleFilterSIR(ParticleFilter):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * propagation and measurement models are largely hardcoded (except for standard deviations.
    """

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm):
        """
        Initialize the SIR particle filter.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param resampling_algorithm: Algorithm that must be used for core.
        """
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Set SIR specific properties
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()

    def needs_resampling(self):
        """
        Method that determines whether not a core step is needed for the current particle filter state estimate.
        The sampling importance core (SIR) scheme resamples every time step hence always return true.

        :return: Boolean indicating whether or not core is needed.
        """
        return True

    #TODO : make measurement a list
    def update(self, plants_motion_move_distance, measurement):
        # Loop over all particles
        new_particles = []
        for par in self.particles:
            # Propagate the particle state according to the current particle
            propagated_state = self.propagate_sample(par[1], plants_motion_move_distance)

            # Compute current particle's weight
            weight = par[0] * self.compute_likelihood(propagated_state, measurement)
            #print("Old and new weight before normalization : {}, {}".format(par[0], weight))

            # Store
            new_particles.append([weight, propagated_state])

        # Update particles
        self.particles = self.normalize_weights(new_particles)

        # Resample if needed
        #if self.needs_resampling():
        #    self.particles = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
