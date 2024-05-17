from abc import abstractmethod
import copy
import numpy as np

# Modified code from :
# Jos Elfring, Elena Torta, and René van de Molengraft.
# Particle filters: A hands-on tutorial.
# Sensors, 21(2), 2021.

class ParticleFilter:
    def __init__(self, number_of_particles, limits, process_noise, measurement_noise):
        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles
        self.particles = []

        # State related settings
        # For the moment we are only tracking the position parameter
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
        # Make sure state does not exceed allowed limits
        # TODO: find a better way of validating states

        while state[0] < self.position_min:
            state[0] += (self.position_max - self.position_min)

        if state[0] > self.position_max:
            state[0] -= (self.position_max - self.position_min)

        return state

    def get_average_state(self):
        """
        Compute average state according to all weighted particles
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
        propagated_sample = copy.deepcopy(sample)

        # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
        motion_move_distance_with_noise = np.random.normal(motion_move_distance, self.process_noise[0], 1)[0]

        # If the new position value is more the height than we move back the particle
        new_pos = propagated_sample[0] + motion_move_distance_with_noise
        if new_pos > self.position_max:
            propagated_sample[0] -= motion_move_distance_with_noise + (new_pos - self.position_max)
        else:
            propagated_sample[0] = new_pos

        # Make sure we stay within cyclic world
        return self.validate_state(propagated_sample)

    # p(zk / xk)
    def compute_likelihood(self, sample, measurement):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks.
        """

        # Initialize measurement likelihood
        likelihood_sample = 1.0

        # Expected measurement assuming the current particle state
        expected_position = sample[0]

        # Map difference true and expected distance measurement to probability
        # Normal distribution
        #TODO : think about Bernouilli distribution
        pr_z_given_position = \
            np.exp(-(expected_position - measurement) * (expected_position - measurement) /
                   (2 * self.measurement_noise[0] * self.measurement_noise[0]))

        #print("Expected position, pr_z_given_position : {}, {}"
        #      .format(expected_position, pr_z_given_position))

        # We will probably need to use multiplication when using the 7 parameters
        likelihood_sample *= pr_z_given_position

        return likelihood_sample

    @abstractmethod
    def update(self, plants_motion_move_distance, measurement):
        """
        Process a measurement given the measured plants displacement. Abstract method that must be implemented in derived
        class.

        """

        pass
