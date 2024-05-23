import numpy as np
import cv2 as cv

from .world import World


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

    def draw_complete_particle(self, offset, position, ir, skew, convergence, ip):

        # Drawing of the particular particle using offset and position
        self.draw_particular_plant(offset, position)

        # Drawing the horizontally neighboring plants to the particular plant
        self.draw_horizontal_neighbors(offset, position, ir)

    def draw_particular_plant(self, offset, position):
        if not (self.is_valid_coordinates(offset, position)):
            print("Error: offset and/or position has an invalid value")
            return -1
        center = np.asarray([offset, position])

        perspective_coef = center[1] / self.world.height
        color = (0, 255, 0)

        cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                      markerSize=int(40 * perspective_coef), thickness=4)

    def draw_horizontal_neighbors(self, offset, position, ir):
        """
        Draws the horizontal neighbours of the particular plant, in other words plants that are located to the left,
        right and at the same height as the particular plant.
        """

        # Drawing parameters
        perspective_coef = position / self.world.height
        color = (255, 0, 0)

        # Drawing left neighbors
        leftOffset = offset - ir

        while self.is_valid_coordinates(leftOffset, position):
            center = np.asarray([leftOffset, position])

            cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                          markerSize=int(40 * perspective_coef), thickness=4)

            leftOffset -= ir

        # Drawing right neighbors
        rightOffset = offset + ir

        while self.is_valid_coordinates(rightOffset, position):
            center = np.asarray([rightOffset, position])

            cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                          markerSize=int(40 * perspective_coef), thickness=4)

            rightOffset += ir

    def draw_top_particular_plant(self, offset, position, skew):
        return True


    def is_valid_coordinates(self, x, y):
        """
        Returns a boolean indicating whether (x, y) is inside the image or not (valid position or not).
        """
        if x < 0 or x > self.world.width:
            return False

        if y < 0 or y > self.world.height:
            return False

        return True

    def draw(self, plants, particles, n_particles):
        # Empty image
        self.img = np.zeros((self.world.height, self.world.width, 3), np.uint8)

        # Draw plants and particles
        self.draw_plants(plants)
        self.draw_particles(particles, n_particles)

        # Testing of the function draw_complete_particle
        #self.img = np.zeros((self.world.height, self.world.width, 3), np.uint8)

        offset = 250
        position = self.world.height - 40
        ir = 70

        #self.draw_complete_particle(offset, position, ir, -1, -1, -1)
