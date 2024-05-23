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

        # Drawing of the particular plant using offset and position
        self.draw_particular_plant(offset, position)

        # Getting bottom plants coordinates and the number of right and left plants
        bottom_plants, nb_left_plants, nb_right_plants = self.get_bottom_plants(offset, position, ir)

        # Drawing the plants located at the bottom
        perspective_coef = position / self.world.height
        color = (255, 0, 0)

        for center in bottom_plants:
            cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                          markerSize=int(40 * perspective_coef), thickness=4)

        # Getting the crossing point between the row where the particular plant is and the top of the image.
        top_particular_crossing_point = self.get_top_crossing_point(offset, position, skew)

        # Getting the crossing point for each row : point of intersection between a row and the top of the image.
        ir_top = convergence * ir
        top_crossing_points = self.get_top_plants(top_particular_crossing_point[0], top_particular_crossing_point[1],
                                                  ir_top, nb_left_plants, nb_right_plants)

        # Drawing all the remaining plants
        for center in top_crossing_points:
            cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                          markerSize=int(40 * perspective_coef), thickness=4)

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
        neighbours of the particular plant. Returns also the number of plants present to its left and the number of
        plants present to its right.
        The first nb_lefts_plants-1 coordinates of the returned list correspond to those of the left plants and the
        remaining correspond to the coordinates of the right plants
        """
        # List of coordinates to be returned
        horizontal_neighbors = []

        # Number of left and right neighbors
        nb_left_neighbours = 0
        nb_right_neighbours = 0

        # Appending the left neighbors of the particular plant
        leftOffset = offset - ir
        while self.are_coordinates_valid(leftOffset, position):
            center = np.asarray([leftOffset, position])
            horizontal_neighbors.append(center)
            nb_left_neighbours += 1
            leftOffset -= ir

        # Appending the particular plant
        center = np.asarray([offset, position])
        horizontal_neighbors.append(center)

        # Appending right neighbors of the particular plant
        rightOffset = offset + ir
        while self.are_coordinates_valid(rightOffset, position):
            center = np.asarray([rightOffset, position])
            horizontal_neighbors.append(center)
            nb_right_neighbours += 1
            rightOffset += ir

        return (horizontal_neighbors, nb_left_neighbours, nb_right_neighbours)

    def get_top_crossing_point(self, bottom_plant_x, bottom_plant_y, skew):
        """
        Returns the coordinates of the point located at the top of the image (height = 0) regarding the skew angle of
        the crop rows and the coordinates of a plant located at the bottom.
        """
        # Use of the TOA formula
        x_coordinate = np.tan(skew) * bottom_plant_y + bottom_plant_x
        center = np.asarray([int(x_coordinate), 0])

        # cv.line(self.img, (bottom_plant_x, bottom_plant_y), center, 255, 1)
        # cv.line(self.img, (bottom_plant_x, bottom_plant_y), (250, 0), 255, 1)
        # cv.line(self.img, (bottom_plant_x, bottom_plant_y), (500, bottom_plant_y), 255, 1)

        return center

    def get_top_plants(self, top_offset, top_position, ir_at_top, nb_left_neighbors, nb_right_neighbors):
        """
        Returns the coordinates of the plants that are located in the top crop row.

        Remarque : we purposefully don't check if the coordinates are valid coordinates (valid in the sense in
        the image).
        """
        # List of coordinates to be returned
        horizontal_neighbors = []

        # Number of plants that have been treated (in other word, while loop variable)
        nb_plants_treated = 0

        # Appending the left plants
        left_plant_x = top_offset
        while nb_plants_treated < nb_left_neighbors:
            left_plant_x -= ir_at_top
            center = np.asarray([left_plant_x, top_position])
            horizontal_neighbors.append(center)
            nb_plants_treated += 1

        # Appending the top particular plant
        center = np.asarray([top_offset, top_position])
        horizontal_neighbors.append(center)
        nb_plants_treated += 1

        # Appending the right plants
        right_plant_x = top_offset
        while nb_plants_treated < (nb_left_neighbors + nb_right_neighbors + 1):
            right_plant_x += ir_at_top
            center = np.asarray([right_plant_x, top_position])
            horizontal_neighbors.append(center)
            nb_plants_treated += 1

        return horizontal_neighbors

    def get_row_plants(self):
        """
        Takes a plant at the bottom of the image and its row's crossing point with the top of the image.
        Returns the coordinates of all the plants located in that row and that are within the image.
        """

        return True

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
        skew = np.pi / 34
        # skew = 0

        self.draw_complete_particle(offset, position, ir, skew, -1, -1)
