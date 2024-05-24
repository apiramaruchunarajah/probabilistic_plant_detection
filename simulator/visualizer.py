import numpy as np
import cv2 as cv

from .world import World
from .particle import Particle


# Visualizer draws plants and particles
class Visualizer:
    def __init__(self, world):
        self.world = world
        self.img = np.zeros((world.height, world.width), np.uint8)

    def draw_plants(self, plants):
        for row_idx in range(plants.nb_rows):
            plant_positions, plant_types = plants.getPlantsToDraw(row_idx)
            # Draw
            for i, center in enumerate(plant_positions):
                if (center[0] < 0) or (center[1] < 0) or (center[0] > self.world.width) or (
                        center[1] > self.world.height):
                    # nb_visible_plants = nb_visible_plants - 1
                    pass
                else:
                    perspective_coef = center[1] / self.world.height
                    if plant_types[i] == 0:
                        cv.circle(self.img, center, int(20 * perspective_coef), (255, 255, 255), -1)
                    elif plant_types[i] == 1:
                        cv.drawMarker(self.img, center, (255, 255, 255), markerType=cv.MARKER_CROSS,
                                      markerSize=int(50 * perspective_coef), thickness=5)
                    elif plant_types[i] == 2:
                        cv.drawMarker(self.img, center, (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS,
                                      markerSize=int(50 * perspective_coef), thickness=5)
                    else:
                        cv.drawMarker(self.img, center, (255, 255, 255), markerType=cv.MARKER_STAR,
                                      markerSize=int(50 * perspective_coef), thickness=5)

    def draw_particles(self, particles, n):
        for i in range(n):
            # Coordinates of the particle
            center = np.asarray([int(self.world.width / 2), int(particles[i][1][0])])

            perspective_coef = center[1] / self.world.height

            # Color of the particle in fonction of the its weight
            if particles[i][0] > 0.70:
                color = (0, 0, 255)
                thickness = 5
            elif particles[i][0] > 0.30:
                color = (255, 0, 0)
                thickness = 4
            else:
                color = (0, 255, 0)
                thickness = 2

            cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                          markerSize=int((7 * thickness) * perspective_coef), thickness=thickness)

    def draw_complete_particle(self, particle):

        # Getting bottom plants coordinates and the number of right and left plants
        bottom_plants, nb_left_plants, nb_right_plants = particle.get_bottom_plants()

        # Drawing parameters
        color = (255, 255, 255)
        markerType = cv.MARKER_STAR

        # Drawing the plants located at the bottom
        perspective_coef = particle.position / self.world.height

        for center in bottom_plants:
            cv.drawMarker(self.img, center, color, markerType=markerType,
                          markerSize=int(40 * perspective_coef), thickness=4)

        # Getting the crossing point between the row where the particular plant is and the top of the image.
        top_particular_crossing_point = particle.get_particular_plant_top_crossing_point()

        # Getting the crossing point for each row : point of intersection between a row and the top of the image.
        top_crossing_points = particle.get_all_top_crossing_points(top_particular_crossing_point[0],
                                                                   top_particular_crossing_point[1], nb_left_plants,
                                                                   nb_right_plants)

        if len(bottom_plants) != len(top_crossing_points):
            print("Error: number of bottom plants and number of top crossing points are not equal.")
            return -1

        # Getting the vanishing point
        # We could take any two plants to compute the vanishing point, here plant 0 and 1
        if len(bottom_plants) < 2:
            print("Can not compute the vanishing point.")
            return -1

        vanishing_point = particle.get_vanishing_point(bottom_plants[0], top_crossing_points[0],
                                                       bottom_plants[1], top_crossing_points[1])

        print("Vanishing point1 : {}".format(vanishing_point))

        # Drawing all the remaining plants row by row.
        # For each row
        for i in range(len(bottom_plants)):
            bottom_plant = bottom_plants[i]

            # Getting the coordinates of the plants located in the row
            row_plants = particle.get_row_plants(bottom_plant, vanishing_point)

            # Drawing all the plants located in the row
            for center in row_plants:
                perspective_coef = center[1] / self.world.height
                cv.drawMarker(self.img, center, color, markerType=markerType,
                              markerSize=int(40 * perspective_coef), thickness=4)

    def draw(self, plants, particles, n_particles):
        # Empty image
        self.img = np.zeros((self.world.height, self.world.width, 3), np.uint8)

        # Draw plants and particles
        self.draw_plants(plants)
        self.draw_particles(particles, n_particles)

        # Testing of the function draw_complete_particle
        self.img = np.zeros((self.world.height, self.world.width, 3), np.uint8)

        offset = 240
        position = self.world.height - 40
        ir = 110
        skew = np.pi / 34
        # skew = 0
        convergence = 60 / ir
        ip = 70

        particle = Particle(self.world, offset, position, ir, ip, convergence, skew)

        self.draw_complete_particle(particle)
