import numpy as np
import cv2 as cv

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

            # We consider now binary images
            color = (255, 255, 255)
            cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                          markerSize=int((7 * thickness) * perspective_coef), thickness=thickness)

    def draw_complete_particle(self, particle):

        # Getting the coordinates of every plant to draw
        plants = particle.get_all_plants()

        # Method using loi normale centrée réduite et évaluée sur la distance
        # # Draw all the plants
        # for plant in plants:
        #     # Draw an heat map around the plant position
        #     for x in range(int(plant[0] - width/2), int(plant[0] + width/2)):
        #         for y in range(int(plant[1] - height/2), int(plant[1] + height/2)):
        #
        #             if self.world.are_coordinates_valid(x, y):
        #                 # Distance between (x, y) and the plant position
        #                 distance = np.sqrt(np.square(x - plant[0]) + np.square(y - plant[1]))
        #
        #                 # Use of the normal law (centrée réduite)
        #                 pr = (1 / np.sqrt(2 * np.pi)) * np.exp(- np.square(distance) / 2)
        #
        #                 # Matching of the intensity of the (x, y) pixel according to the above probability
        #                 intensity = max_intensity * pr
        #                 s += pr
        #
        #                 if pr > 0.7:
        #                     print("pr, intensity : {}, {}".format(pr, intensity))
        #
        #                 self.img[y][x] = intensity

        # Radius of the square surrounding the central plant position pixel
        radius = 40

        # Different grey levels representing pixels around a plant position
        min_intensity = 40
        max_intensity = 255
        intensities = np.arange(min_intensity, max_intensity, int((max_intensity - min_intensity) / radius))
        nb_intensities = len(intensities)

        # # Drawing all the plants
        # for center in plants:
        #     # Perspective coefficient, further a plant is, smaller it is
        #     perspective_coef = center[1] / self.world.height
        #
        #     if self.world.are_coordinates_valid(center[0], center[1]):
        #         # Drawing the center with full intensity
        #         self.img[center[1]][center[0]] = max_intensity
        #
        #         # Square represents the different "square pixels" surrounding the plant pixel
        #         for square in range(1, radius):
        #             # Drawing the top line pixels
        #             y = center[1] - square
        #             for x in range(center[0] - radius, center[0] + radius):
        #                 if self.world.are_coordinates_valid(x, y):
        #                     self.img[y][x] = intensities[nb_intensities - square]
        #
        #             # Drawing the bottom line pixels
        #             y = center[1] + square
        #             for x in range(center[0] - radius, center[0] + radius):
        #                 if self.world.are_coordinates_valid(x, y):
        #                     self.img[y][x] = intensities[nb_intensities - square]
        #
        #             # Drawing the middle line pixels
        #             y = center[1]

        for center in plants:
            # Drawing parameters
            intensity = 255
            radius = 6

            cv.circle(self.img, center, radius, intensity, -1)

    def draw(self, plants, particles, n_particles):
        # Empty image
        self.img = np.zeros((self.world.height, self.world.width), np.uint8)

        # Draw plants and particles
        self.draw_plants(plants)
        #self.draw_particles(particles, n_particles)

        # Testing of the function draw_complete_particle
        #self.img = np.zeros((self.world.height, self.world.width, 1), np.uint8)

        offset = 240
        position = self.world.height - 40
        ir = 110  # Problematic when go to a low ir (~40 for example)
        skew = -np.pi / 34
        convergence = 0.04
        ip = 110

        particle = Particle(self.world, offset, position, ir, ip, convergence, skew)

        #self.draw_complete_particle(particle)

    def measure(self):
        """
        Looks at its image and returns the parameters offset, position, convergence, ....
        """
        self.get_particular_plant()
        return True

    def get_particular_plant(self):
        green_pixels = self.get_green_pixels()
        print("Green pixels : {}".format(len(green_pixels)))
        return True

    def get_green_pixels(self):
        return np.where(self.img != 0)
