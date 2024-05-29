#!/usr/bin/env python
import cv2 as cv

import numpy as np

# Simulation + plotting requires plants, visualizer and world
from simulator import Plants, Visualizer, World

from simulator.particle import Particle

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle0 filters)
from core.resampling.resampler import ResamplingAlgorithms

# Particle filters
from core.particle_filters.particle_filter_sir import ParticleFilterSIR

if __name__ == '__main__':

    # Initialize world
    world = World(500, 700, 10)

    # Number of simulated time steps
    n_time_steps = 30 + 40

    # Initialize visualizer
    visualizer = Visualizer(world)

    ##
    # True plants properties (simulator settings)
    ##

    # Setpoint (desired) motion speed
    plants_setpoint_motion_move_distance = 11

    # True simulated plants motion is set point plus additive zero mean Gaussian noise with these standard deviation
    # Plants move at speed +11 +- std
    # For the moment de movement has no noise
    true_plants_motion_move_distance_std = 0

    # Plants measurements are corrupted by measurement noise
    true_plants_meas_noise_position_std = 7

    # Size of a plant : length of the side of a square
    plant_size = 6

    # Initialize plants
    plants = Plants(world, -100, 250, 80, 110, o=0, nb_rows=7, nb_plant_types=4)
    plants.setStandardDeviations(true_plants_motion_move_distance_std, true_plants_meas_noise_position_std)
    plants.generate_plants()

    ##
    # Particle filter settings
    ##

    number_of_particles = 1
    # Limit values for the parameters we track.
    pf_state_limits = [0, world.width,  # Offset
                       world.height - 240, world.height,  # Position
                       world.height / 6, world.height / 2,  # Inter-plant
                       world.width / 6, world.width / 4,  # Inter-row
                       np.pi / 17, np.pi / 6,  # Skew, limits have to be considered in negative and positive values.
                       0, 1]  # Convergence

    # Process model noise (zero mean additive Gaussian noise)
    # This noise has a huge impact on the correctness of the particle0 filter
    motion_model_move_distance_std = 40
    process_noise = [40,  # Offset
                     motion_model_move_distance_std,  # Position
                     110,  # Inter-plant
                     110,  # Inter-row
                     np.pi / 256,  # Skew
                     0.11]  # Convergence

    # Probability associated to the measurement image. We have the probability for a pixel
    probability_in = 0.8
    probability_out = 0.02
    measurement_uncertainty = [probability_in, probability_out]

    # Set resampling algorithm used
    # TODO: compare with the other resampling algorithms
    algorithm = ResamplingAlgorithms.STRATIFIED

    # Initialize SIR particle0 filter: resample every time step
    particle_filter_sir = ParticleFilterSIR(
        world=world,
        number_of_particles=number_of_particles,
        limits=pf_state_limits,
        process_noise=process_noise,
        measurement_noise=measurement_uncertainty,
        resampling_algorithm=algorithm)

    # Particles are selected uniformly randomly
    particle_filter_sir.initialize_particles_uniform()

    particle_filter_sir.print_particles()

    ##
    # Start simulation
    ##
    max_weights = []
    for i in range(n_time_steps):
        # Simulate plants motion (required motion will not exactly be achieved)
        plants.move(plants_setpoint_motion_move_distance)

        # Simulate measurement
        meas_image = visualizer.measure()

        # Update SIR particle filter
        particle_filter_sir.update(plants_setpoint_motion_move_distance, meas_image, plant_size)

        # # Show maximum normalized particle0 weight (converges to 1.0) and correctness (0 = correct)
        # w_max = particle_filter_sir.get_max_weight()
        # max_weights.append(w_max)
        # # Distance between the measured value and the average particle0 value
        # correctness = np.sqrt(np.square(particle_filter_sir.get_average_state() - meas_position))
        # print("Time step {}: max weight: {}, correctness: {}".format(i, w_max, correctness))

        # Visualization
        # Drawing plants
        visualizer.draw(plants, particle_filter_sir.particles, particle_filter_sir.n_particles)

        # Drawing a particle
        avg_state = particle_filter_sir.get_average_state()
        avg_particle = Particle(world, avg_state[0], avg_state[1], avg_state[2], avg_state[3],
                                avg_state[4], avg_state[5])
        visualizer.draw_complete_particle(avg_particle)

        # # Drawing the first particle
        # state = particle_filter_sir.particles[0][1]
        # particle = Particle(world, state[0], state[1], state[2], state[3],
        #                     state[4], state[5])
        # visualizer.draw_complete_particle(particle)

        # Showing the image
        cv.imshow("Crop rows", visualizer.img)
        cv.waitKey(0)

    # Print Degeneracy problem
    # Plot weights as function of time step
    #fontSize = 14
    #plt.rcParams.update({'font.size': fontSize})
    #plt.plot(range(n_time_steps), max_weights, 'k')
    #plt.xlabel("Time index")
    #plt.ylabel("Maximum particle0 weight")
    #plt.xlim(0, n_time_steps - 1)
    #plt.ylim(0, 1.1)
    #plt.show()
