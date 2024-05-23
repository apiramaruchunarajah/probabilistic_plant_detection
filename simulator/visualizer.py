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

        # Drawing the plants located at the bottom
        bottom_plants = self.get_bottom_plants(offset, position, ir)

        perspective_coef = position / self.world.height
        color = (255, 0, 0)

        for center in bottom_plants:
            cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                          markerSize=int(40 * perspective_coef), thickness=4)

        # Getting the coordinates of the top particular plant
        top_particular_plant = self.get_top_crossing_point(offset, position, skew)

        # Drawing the plants located at the top

    def draw_particular_plant(self, offset, position):
        """
        Draws the particular plant.
        """
        if not (self.are_coordinates_valid(offset, position)):
            print("Error: offset and/or position has an invalid value")
            return -1

        center = np.asarray([offset, position])

        perspective_coef = center[1] / self.world.height
        color = (0, 255, 0)

        cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                      markerSize=int(40 * perspective_coef), thickness=4)

    def get_bottom_plants(self, offset, position, ir):
        """
        Returns the coordinates of the plants that are located in the bottom crop row, in other words the horizontal
        neighbours of the particular plant.
        """
        # List of coordinates to be returned
        horizontal_neighbors = []

        # Appending the left neighbors of the particular plant
        leftOffset = offset - ir
        while self.are_coordinates_valid(leftOffset, position):
            center = np.asarray([leftOffset, position])
            horizontal_neighbors.append(center)
            leftOffset -= ir

        # Appending right neighbors of the particular plant
        rightOffset = offset + ir
        while self.are_coordinates_valid(rightOffset, position):
            center = np.asarray([rightOffset, position])
            horizontal_neighbors.append(center)
            rightOffset += ir

        return horizontal_neighbors

    def get_top_crossing_point(self, bottom_plant_x, bottom_plant_y, skew):
        """
        Returns the coordinates of the point located at the top of the image (height = 0) regarding the skew angle of
        the crop rows and the coordinates of a plant located at the bottom.
        """
        x_coordinate = np.tan(skew) * bottom_plant_y + bottom_plant_x
        center = np.asarray([int(x_coordinate), 0])

        cv.line(self.img, (bottom_plant_x, bottom_plant_y), center, 255, 1)
        # cv.line(self.img, (bottom_plant_x, bottom_plant_y), (250, 0), 255, 1)
        # cv.line(self.img, (bottom_plant_x, bottom_plant_y), (500, bottom_plant_y), 255, 1)

        return center

    def are_coordinates_valid(self, x, y):
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
        self.img = np.zeros((self.world.height, self.world.width, 3), np.uint8)

        offset = 250
        position = self.world.height - 40
        ir = 70
        skew = np.pi / 24

        self.draw_complete_particle(offset, position, ir, skew, -1, -1)
