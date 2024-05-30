import numpy as np
import cv2 as cv

from .particle import Particle


# Visualizer draws plants and particles
class Visualizer:
    def __init__(self, world):
        self.world = world
        self.img = np.zeros((world.height, world.width, 3), np.uint8)

    def draw_plants(self, plants):
        # Green color for plants
        color = (0, 255, 0)

        # For each row
        for row_idx in range(plants.nb_rows):
            plant_positions, plant_types = plants.getPlantsToDraw(row_idx)
            # Draw
            for i, center in enumerate(plant_positions):
                if self.world.are_coordinates_valid(center[0], center[1]):
                    perspective_coef = center[1] / self.world.height
                    cv.circle(self.img, center, int(20 * perspective_coef), color, -1)
                    # if plant_types[i] == 0:
                    #     cv.circle(self.img, center, int(20 * perspective_coef), color, -1)
                    # elif plant_types[i] == 1:
                    #     cv.drawMarker(self.img, center, color, markerType=cv.MARKER_CROSS,
                    #                   markerSize=int(50 * perspective_coef), thickness=5)
                    # elif plant_types[i] == 2:
                    #     cv.drawMarker(self.img, center, color, markerType=cv.MARKER_TILTED_CROSS,
                    #                   markerSize=int(50 * perspective_coef), thickness=5)
                    # else:
                    #     cv.drawMarker(self.img, center, color, markerType=cv.MARKER_STAR,
                    #                   markerSize=int(50 * perspective_coef), thickness=5)

    def draw_particles(self, particles, n):
        for i in range(n):
            # Coordinates of the particle0
            center = np.asarray([int(self.world.width / 2), int(particles[i][1][0])])

            perspective_coef = center[1] / self.world.height

            # Color of the particle0 in fonction of the its weight
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
        
        if plants == -1:
            print("Visualizer: Can not draw the particle0")
            return
        
        # Drawing every plant
        for center in plants:
            # Drawing parameters
            color = (255, 0, 0)
            radius = 6
        
            try:
                cv.circle(self.img, (int(center[0]), int(center[1])), radius, color, -1)
            except:
                print("Problematic center : {}".format(center))

        # Drawing parameters
        # color = (255, 0, 0)
        # radius = 6
        #
        # center = np.asarray([particle.offset, particle.position])
        # cv.circle(self.img, (int(center[0]), int(center[1])), radius, color, -1)

    def draw(self, plants, particles, n_particles):
        # Empty image
        self.img = np.zeros((self.world.height, self.world.width, 3), np.uint8)

        # Draw plants and particles
        self.draw_plants(plants)

        # # Draw lines to help debugging
        # cv.line(self.img, (0, 200), (499, 200), (255, 255, 255))
        # cv.line(self.img, (0, 400), (499, 400), (255, 255, 255))
        # cv.line(self.img, (0, 600), (499, 600), (255, 255, 255))
        # cv.line(self.img, (0, 100), (499, 100), (255, 255, 255))
        # cv.line(self.img, (0, 300), (499, 300), (255, 255, 255))
        # cv.line(self.img, (0, 500), (499, 500), (255, 255, 255))
        #
        # cv.line(self.img, (200, 0), (200, 699), (255, 255, 255))
        # cv.line(self.img, (400, 0), (400, 699), (255, 255, 255))
        # cv.line(self.img, (100, 0), (100, 699), (255, 255, 255))
        # cv.line(self.img, (300, 0), (300, 699), (255, 255, 255))

        # Testing of the function draw_complete_particle
        # self.img = np.zeros((self.world.height, self.world.width, 1), np.uint8)
        #
        # offset = 240
        # position = self.world.height - 40
        # ir = 110  # Problematic when go to a low ir (~40 for example)self.get_particular_plant()
        # skew = -np.pi / 34
        # convergence = 0.04
        # ip = 110
        #
        # particle = Particle(self.world, offset, position, ir, ip, convergence, skew)
        #
        # self.draw_complete_particle(particle0)

    def measure(self):
        """
        Returns its image
        /!Looks at its image and returns the parameters offset, position, convergence, ....!\
        """
        return self.img
