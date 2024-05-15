from abc import abstractmethod
import copy
import numpy as np


class ParticleFilter:
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * Abstract class
    """

    def __init__(self, number_of_particles, limits, process_noise, measurement_noise):
        """
        Initialize the abstract particle filter.

        :param number_of_particles: Number of particles
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax]
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular]
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle]
        """

        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles
        self.particles = []

        # State related settings
        self.state_dimension = 1
        #self.inter_plant_min = limits[0]
        #self.inter_plant_max = limits[1]
        #self.inter_row_min = limits[2]
        #self.inter_row_max = limits[3]
        #self.skew_min = limits[4]
        #self.skew_max = limits[5]
        #self.convergence_min = limits[6]
        #self.convergence_max = limits[7]
        #self.offset_min = limits[8]
        #self.offset_max = limits[9]
        self.position_min = limits[0]
        self.position_max = limits[1]
        #self.speed_min = limits[12]
        #self.speed_max = limits[13]

        # Set noise
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, heading). No arguments are required
        and function always succeeds hence no return value.
        """

        # Initialize particles with uniform weight distribution
        self.particles = []
        weight = 1.0 / self.n_particles
        for i in range(self.n_particles):
            # Add particle i
            self.particles.append(
                [weight, [np.random.uniform(self.position_min, self.position_max, 1)[0]]]
            )

        print("Initial particles position value :")
        self.print_particles()

    def validate_state(self, state):
        """
        Validate the state. State values outide allowed ranges will be corrected for assuming a 'cyclic world'.

        :param state: Input particle state.
        :return: Validated particle state.
        """

        # Make sure state does not exceed allowed limits
        # TODO: find a better way of validating states

        return state

        if state[0] > self.position_max:
            state[0] = self.position_min

        if state[0] < self.position_min:
            state[0] = self.position_max

        return state

    def set_particles(self, particles):
        """
        Initialize the particle filter using the given set of particles.

        :param particles: Initial particle set: [[weight_1, [x1, y1, theta1]], ..., [weight_n, [xn, yn, thetan]]]
        """

        # Assumption: particle have correct format, set particles
        self.particles = copy.deepcopy(particles)
        self.n_particles = len(self.particles)

    def get_average_state(self):
        """
        Compute average state according to all weighted particles

        :return: Average x-position, y-position and orientation
        """

        # Compute sum of all weights
        sum_weights = 0.0
        for weighted_sample in self.particles:
            sum_weights += weighted_sample[0]

        # Compute weighted average
        avg_position = 0.0
        for weighted_sample in self.particles:
            avg_position += weighted_sample[0] / sum_weights * weighted_sample[1][0]

        return avg_position

    def get_max_weight(self):
        """
        Find maximum weight in particle filter.

        :return: Maximum particle weight
        """
        return max([weighted_sample[0] for weighted_sample in self.particles])

    def print_particles(self):
        """
        Print all particles: index, state and weight.
        """

        print("Particles:")
        for i in range(self.n_particles):
            print(" ({}): {} with w: {}%".format(i + 1, self.particles[i][1], self.particles[i][0]*100))

    @staticmethod
    def normalize_weights(weighted_samples):
        """
        Normalize all particle weights.
        """

        # Compute sum weighted samples
        sum_weights = 0.0
        for weighted_sample in weighted_samples:
            sum_weights += weighted_sample[0]

        # Check if weights are non-zero
        if sum_weights < 1e-15:
            print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(
                sum_weights))

            # Set uniform weights
            return [[1.0 / len(weighted_samples), weighted_sample[1]] for weighted_sample in weighted_samples]

        # Return normalized weights
        return [[weighted_sample[0] / sum_weights, weighted_sample[1]] for weighted_sample in weighted_samples]

    # p(xk | xk-1)
    def propagate_sample(self, sample, motion_move_distance):
        """
        Propagate an individual sample with a simple motion model that assumes the robot rotates angular_motion rad and
        then moves forward_motion meters in the direction of its heading. Return the propagated sample (leave input
        unchanged).

        :param motion_move_distance: Plants' move distance
        :param sample: Sample (unweighted particle) that must be propagated
        :return: propagated sample
        """

        propagated_sample = copy.deepcopy(sample)

        # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
        motion_move_distance_with_noise = np.random.normal(motion_move_distance, self.process_noise[0], 1)[0]

        # Move the position of the particle
        # Special case where a particle leaves the image : the particle is being put back -motion_move_distance behind
        if propagated_sample[0]+motion_move_distance_with_noise > self.position_max:
            propagated_sample[0] -= motion_move_distance  # or motion_move_distance_with_noise?
        else:
            propagated_sample[0] += motion_move_distance_with_noise

        # Make sure we stay within cyclic world
        return self.validate_state(propagated_sample)

    def compute_likelihood(self, sample, measurement):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks.

        :param sample: Sample (unweighted particle) that must be propagated
        :param measurement: For the moment is not a list #TODO : make it a list
        :return Likelihood
        """

        # Initialize measurement likelihood
        likelihood_sample = 1.0

        # Expected measurement assuming the current particle state
        expected_position = sample[0]

        # Map difference true and expected distance measurement to probability
        # Normal distribution #TODO : think about Bernouilli distribution
        pr_z_given_position = \
            np.exp(-(expected_position - measurement) * (expected_position - measurement) /
                   (2 * self.measurement_noise[0] * self.measurement_noise[0]))

        #print("Expected position, pr_z_given_position : {}, {}"
        #      .format(expected_position, pr_z_given_position))

        # We will probably need to use multiplication when using the 7 parameters
        likelihood_sample *= pr_z_given_position

        # Return importance weight based on all landmarks
        return likelihood_sample

    @abstractmethod
    def update(self, plants_motion_move_distance, measurement):
        """
        Process a measurement given the measured robot displacement. Abstract method that must be implemented in derived
        class.

        """

        pass
