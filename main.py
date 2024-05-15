#!/usr/bin/env python
import cv2 as cv

# Simulation + plotting requires a robot, visualizer and world
from simulator import Plants, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling.resampler import ResamplingAlgorithms

# Particle filters
from core.particle_filters.particle_filter_sir import ParticleFilterSIR

if __name__ == '__main__':

    print("Running example particle filter demo.")

    ##
    # Set simulated world and visualization properties
    ##
    # Initialize world
    world = World(500, 700, 10)

    # Number of simulated time steps
    n_time_steps = 30+40

    # Initialize visualizer
    visualizer = Visualizer(world)

    ##
    # True plants properties (simulator settings)
    ##

    # Setpoint (desired) motion speed
    plants_setpoint_motion_move_distance = 11

    # True simulated plants motion is set point plus additive zero mean Gaussian noise with these standard deviation
    # Plants move at speed +11 +- std
    true_plants_motion_move_distance_std = 0

    # Plants measurements are corrupted by measurement noise
    true_plants_meas_noise_position_std = 0

    # Initialize plants
    plants = Plants(world, -500, 250, 80, 110, o=0, s=0, c=0, nb_rows=7, nb_plant_types=4)
    plants.setStandardDeviations(true_plants_motion_move_distance_std, true_plants_meas_noise_position_std)
    plants.generate_plants()

    ##
    # Particle filter settings
    ##

    number_of_particles = 40
    # For the moment we track position, which is between 0 and height
    pf_state_limits = [0, world.height]

    # Process model noise (zero mean additive Gaussian noise)
    motion_model_move_distance_std = 0
    process_noise = [motion_model_move_distance_std]

    # Measurement noise (zero mean additive Gaussian noise)
    meas_model_position_std = 7
    measurement_noise = [meas_model_position_std]

    # Set resampling algorithm used
    algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Initialize SIR particle filter: resample every time step
    particle_filter_sir = ParticleFilterSIR(
        number_of_particles=number_of_particles,
        limits=pf_state_limits,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        resampling_algorithm=algorithm)
    particle_filter_sir.initialize_particles_uniform()

    ##
    # Start simulation
    ##
    for i in range(n_time_steps):

        # Simulate plants motion (required motion will not excatly be achieved)
        plants.move(plants_setpoint_motion_move_distance)

        # Simulate measurement
        meas_position = plants.measure()

        # Update SIR particle filter
        particle_filter_sir.update(plants_setpoint_motion_move_distance, meas_position)
        print("Particles after update :")
        particle_filter_sir.print_particles()

        print("Average, Max weight : {}, {}%".format(particle_filter_sir.get_average_state(),
                                                     particle_filter_sir.get_max_weight()*100))

        # Visualization
        visualizer.draw(plants, particle_filter_sir)
        cv.imshow("Crop rows", visualizer.img)
        cv.waitKey(0)