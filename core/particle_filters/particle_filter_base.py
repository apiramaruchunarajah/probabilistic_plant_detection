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
            print("Warning: initializing particle0 filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles
        self.particles = []

        # State related settings
        # For the moment we are not considering the speed parameter.
        self.state_dimension = 6
        self.offset_min = limits[0]
        self.offset_max = limits[1]
        self.position_min = limits[2]
        self.position_max = limits[3]
        self.inter_plant_min = limits[4]
        self.inter_plant_max = limits[5]
        self.inter_row_min = limits[6]
        self.inter_row_max = limits[7]
        self.skew_min = limits[8]
        self.skew_max = limits[9]
        self.convergence_min = limits[10]
        self.convergence_max = limits[11]
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
            # Selecting randomly uniformly the parameters' values
            offset = np.random.uniform(self.offset_min, self.offset_max, 1)[0]
            position = np.random.uniform(self.position_min, self.position_max, 1)[0]
            inter_plant = np.random.uniform(self.inter_plant_min, self.inter_plant_max, 1)[0]
            inter_row = np.random.uniform(self.inter_row_min, self.inter_row_max, 1)[0]
            skew = np.random.uniform(self.skew_min, self.skew_max, 1)[0]
            convergence = np.random.uniform(self.convergence_min, self.convergence_max, 1)[0]

            # Add particle0 i
            self.particles.append(
                [weight, [offset, position, inter_plant, inter_row, skew, convergence]]
            )

        print("Initial particles position value :")
        self.print_particles()

    def validate_state(self, state):
        # Make sure state does not exceed allowed limits
        # TODO: find a better way of validating states

        while state[1] < self.position_min:
            state[1] += (self.position_max - self.position_min)

        if state[1] > self.position_max:
            state[1] -= (self.position_max - self.position_min)

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
        avg_offset = 0.0
        avg_position = 0.0
        avg_inter_plant = 0.0
        avg_inter_row = 0.0
        avg_skew = 0.0
        avg_convergence = 0.0
        for weighted_sample in self.particles:
            avg_offset += weighted_sample[0] / sum_weights * weighted_sample[1][0]
            avg_position += weighted_sample[0] / sum_weights * weighted_sample[1][1]
            avg_inter_plant += weighted_sample[0] / sum_weights * weighted_sample[1][2]
            avg_inter_row += weighted_sample[0] / sum_weights * weighted_sample[1][3]
            avg_skew += weighted_sample[0] / sum_weights * weighted_sample[1][4]
            avg_convergence += weighted_sample[0] / sum_weights * weighted_sample[1][5]

        return [avg_offset, avg_position, avg_inter_plant, avg_inter_row, avg_skew, avg_convergence]

    def get_max_weight(self):
        """
        Find maximum weight in particle0 filter.

        :return: Maximum particle0 weight
        """
        return max([weighted_sample[0] for weighted_sample in self.particles])

    def print_particles(self):
        """
        Print all particles: index, state and weight.
        """

        print("Particles:")
        for i in range(self.n_particles):
            print(" ({}): {} with w: {}%".format(i + 1, self.particles[i][1], self.particles[i][0] * 100))

    @staticmethod
    def normalize_weights(weighted_samples):
        """
        Normalize all particle0 weights.
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

    # Motion model
    # ~p(xk | xk-1)
    def propagate_sample(self, sample, motion_move_distance):
        """
        Propagates an individual sample using a motion model where the position is moving downward and the other
        parameters are supposed to stay the same.
        """

        # Copy of the sample we want to propagate.
        propagated_sample = copy.deepcopy(sample)

        # 1. Parameters that are not supposed to be modified
        # Offset, inter-plant, inter-row, skew and convergence are supposed to stay the same. They are equal to their
        # value at the previous time step with some additive zero mean Gaussian noise.
        offset = np.random.normal(propagated_sample[0], self.process_noise[0], 1)[0]
        inter_plant = np.random.normal(propagated_sample[2], self.process_noise[2], 1)[0]
        inter_row = np.random.normal(propagated_sample[3], self.process_noise[3], 1)[0]
        skew = np.random.normal(propagated_sample[4], self.process_noise[4], 1)[0]
        convergence = np.random.normal(propagated_sample[5], self.process_noise[5], 1)[0]

        propagated_sample[0] = offset
        propagated_sample[2] = inter_plant
        propagated_sample[3] = inter_row
        propagated_sample[4] = skew
        propagated_sample[5] = convergence

        # 2. Parameters that are supposed to be modified
        # The value of position for the propagated sample is computed using the forward motion combined with additive
        # zero mean Gaussian noise.
        motion_move_distance_with_noise = np.random.normal(motion_move_distance, self.process_noise[1], 1)[0]

        # If the new position value is more the height than we move back the particle0
        position = propagated_sample[1] + motion_move_distance_with_noise
        if position >= self.position_max:
            propagated_sample[1] -= motion_move_distance_with_noise + (position - self.position_max)
        else:
            propagated_sample[1] = position

        # TODO: modify validate_state
        return self.validate_state(propagated_sample)

    # p(zk / xk)
    def compute_likelihood(self, sample, measurement):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks.
        """

        # Initialize measurement likelihood
        likelihood_sample = 1.0

        # Expected measurement assuming the current particle0 state
        expected_position = sample[0]

        # Map difference true and expected distance measurement to probability
        # Normal distribution
        #TODO : think about Bernoulli distribution
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
        Process a measurement given the measured plants' displacement. Abstract method that must be implemented in
        derived class.

        """

        pass
