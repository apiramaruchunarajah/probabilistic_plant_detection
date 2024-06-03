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

    # np.random.seed(40)

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
    plant_size = 2

    # Area size
    area_size = 4

    # Initialize plants
    plants = Plants(world, -100, 310, 160, 110, o=0, nb_rows=4, nb_plant_types=4)
    plants.setStandardDeviations(true_plants_motion_move_distance_std, true_plants_meas_noise_position_std)
    plants.generate_plants()

    ##
    # Particle filter settings
    ##

    number_of_particles = 40
    # Limit values for the parameters we track.
    pf_state_limits = [0, world.width,  # Offset
                       world.height - 240, world.height,  # Position
                       11, 240,  # Inter-plant, not too small
                       world.width/25, world.width / 4,  # Inter-row, not too low because get_bottom_plants
                                                         # can take too long /!\
                       -np.pi / 8, np.pi / 8,  # Skew
                       0, 0.8]  # Convergence, close to 1 means parallel lines that can cause issues /!\

    # pf_state_limits = [0, world.width,
    #                    0, world.height,
    #                    11, world.height/4,
    #                    world.width/8, world.width/2,  # max 8 rows, min 2 rows
    #                    -np.pi/6, np.pi/6,
    #                    0, 0.8]

    pf_state_limits = [world.width-110, world.width+110,  # Offset
                       world.height - 80, world.height,  # Position, on limite artificiellement l'espace de recherche de position
                       90, 130,  # Inter-plant, not too small
                       151, 170,  # Inter-row, not too low because get_bottom_plants
                       # can take too long /!\          print("Compute likelihood: outside of the considered area.")
                       -np.pi / 12, np.pi / 12,  # Skew
                       0.1, 0.4]  # Convergence, close t                    else:o 1 means parallel lines that can cause issues /!\

    # Process model noise (zero mean additive Gaussian noise)
    # This noise has a huge impact on the correctness of the particle0 filter
    motion_model_move_distance_std = 25
    process_noise = [11,  # Offset
                     motion_model_move_distance_std,  # Position
                     11,  # Inter-plant
                     11,  # Inter-row
                     np.pi / 24,  # Skew
                     0.25]  # Convergence                    
                                                             
    process_noise = [0,  # Offset                            import cv2 as cv
                     (pf_state_limits[3] - pf_state_limits[2]) / 4,  # Position
                     (pf_state_limits[5] - pf_state_limits[4]) / 4,  # IP
                     (pf_state_limits[7] - pf_state_limits[7]) / 4,  # IR
                     (pf_state_limits[9] - pf_state_limits[8]) / 4,  # Skew
                     (pf_state_limits[11] - pf_state_limits[10]) / 2]  # Convergence


    # Probability associated to the measurement image. We have the probability for a pixel
    probability_in = 0.90
    probability_out = 0.01
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

    ##
    # Start simulation
    ##
    max_weights = []
    for i in range(n_time_steps):
        # Simulate plants motion (required motion will not exactly be achieved)
        plants.move(plants_setpoint_motion_move_distance)

        # Visualization
        # Drawing plants
        visualizer.draw(plants, particle_filter_sir.particles, particle_filter_sir.n_particles)

        # Simulate measurement
        meas_image = visualizer.measure()

        # Update SIR particle filter
        particle_filter_sir.update(plants_setpoint_motion_move_distance, meas_image, plant_size, area_size)

        # Show maximum normalized particle weight (converges to 1.0) and correctness (0 = correct)
        w_max = particle_filter_sir.get_max_weight()
        max_weights.append(w_max)
        print("Time step {}: max weight: {}".format(i, w_max))

        # Drawing a particle
        avg_state = particle_filter_sir.get_average_state()
        avg_particle = Particle(world, avg_state[0], avg_state[1], avg_state[2], avg_state[3],
                                avg_state[4], avg_state[5])
        visualizer.draw_complete_particle(avg_particle, (255, 0, 255), 7)
        #print("Avg skew : {}, avg convergence : {}, avg inter-plant : {}"
        #      .format(avg_particle.skew, avg_particle.convergence, avg_particle.ip_at_bottom))

        nb = 0
        # Drawing every particle
        for par in particle_filter_sir.particles:
            # nb += 25
            particle = Particle(world, par[1][0], par[1][1], par[1][2], par[1][3], par[1][4], par[1][5])
            visualizer.draw_complete_particle(particle, (0, nb, 255), 6)

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
