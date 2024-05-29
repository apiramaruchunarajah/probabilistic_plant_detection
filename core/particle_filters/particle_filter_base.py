from abc import abstractmethod
import copy
import numpy as np

# Import of the Particle class
from simulator.particle import Particle


# Modified code from :
# Jos Elfring, Elena Torta, and René van de Molengraft.
# Particle filters: A hands-on tutorial.
# Sensors, 21(2), 2021.

class ParticleFilter:
    def __init__(self, world, number_of_particles, limits, process_noise, measurement_uncertainty):
        if number_of_particles < 1:
            print("Warning: initializing particle0 filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles
        self.particles = []
        self.world = world

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
        self.measurement_uncertainty = measurement_uncertainty

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

            # Add particle i
            self.particles.append(
                [weight, [offset, position, inter_plant, inter_row, skew, convergence]]
            )

        print("Initial particles position value :")
        self.print_particles()

    def validate_state(self, state):
        # Make sure state does not exceed allowed limits

        # Validate Offset
        if state[0] < self.offset_min:
            state[0] = self.offset_min

        if state[0] > self.offset_max:
            state[0] = self.offset_max-1

        # Validate Position
        if state[1] > self.position_max:
            state[1] = self.position_max-1

        if state[1] < self.position_min:
            state[1] = self.position_min

        # Validate Inter-plant
        if state[2] > self.inter_plant_max:
            state[2] = self.inter_plant_max

        if state[2] < self.inter_plant_min:
            state[2] = self.inter_plant_min

        # Validate Inter-Row
        if state[3] > self.inter_row_max:
            state[3] = self.inter_row_max

        if state[3] < self.inter_row_min:
            state[3] = self.inter_row_min

        # Validate skew
        if state[4] < self.skew_min or state[4] > self.skew_max:
            state[4] = (self.skew_max - self.skew_min) / 2

        # Validate convergence:
        if state[5] > self.convergence_max:
            state[5] = self.convergence_max

        if state[5] < self.convergence_min:
            state[5] = self.convergence_min

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

        # If the new position value is more than the height than we move back the particle
        position = propagated_sample[1] + motion_move_distance_with_noise
        if position >= self.position_max:
            propagated_sample[1] -= motion_move_distance_with_noise + (position - self.position_max)
        else:
            propagated_sample[1] = position

        return self.validate_state(propagated_sample)

    # Measurement model
    # p(zk / xk)
    def compute_likelihood(self, sample, measurement, plant_size):
        """
        Compute likelihood p(z|sample) for a specific measurement given (unweighted) sample state.
        The measurement is an image containing plants. The sample is not an image : it contains parameters values from
        which we can draw an image and/or find its plants positions. For each position given by the
        particle, we look at the pixels in the measurement image located around (size of the plant) this position.
        """

        # Expected plant positions assuming the current particle state
        particle = Particle(self.world, sample[0], sample[1], sample[2], sample[3], sample[4], sample[5])
        expected_plant_positions = particle.get_all_plants()

        if expected_plant_positions == -1:
            print("Compute likelihood can't be done because the particle doesn't return a list of plant positions.")
            return 0

        # IN MAIN WE ONLY HAVE ONE PARTICLE FOR THE MOMENT

        # Initialize measurement likelihood for the particle/sample.
        likelihood_sample = 1.0

        # Computing for each expected plant position its probability of really being a position where a plant is.
        for plant in expected_plant_positions:
            # Initializing the total number of pixels in the surrounding and the number of green pixels among them.
            total_nb_surrounding_pixels = 0
            nb_green_pixels = 0

            # Going through all the surrounding pixels.
            for y in range(-int(plant_size / 2), int(plant_size / 2)):
                for x in range(-int(plant_size / 2), int(plant_size / 2)):
                    x_coordinate = int(plant[0] - x)
                    y_coordinate = int(plant[1] - y)

                    if self.world.are_coordinates_valid(x_coordinate, y_coordinate):
                        measured_pixel = measurement[y_coordinate][x_coordinate]
                        total_nb_surrounding_pixels += 1
                        # Checking if the pixel is green.
                        if measured_pixel[1] == 255:
                            nb_green_pixels += 1

            # Getting the number of pixels other than green in that surrounding.
            nb_other_pixels = total_nb_surrounding_pixels - nb_green_pixels

            # Computing the probability associated to the plant position.
            pr_plant_given_position = ((nb_green_pixels * self.measurement_uncertainty[0] +
                                        nb_other_pixels * self.measurement_uncertainty[1])
                                       / total_nb_surrounding_pixels)

            likelihood_sample *= pr_plant_given_position

        return likelihood_sample

    @abstractmethod
    def update(self, plants_motion_move_distance, measurement, plant_size):
        """
        Process a measurement given the measured plants' displacement. Abstract method that must be implemented in
        derived class.

        """

        pass
