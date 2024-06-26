import numpy as np


class Particle:
    def __init__(self, world, offset, position, inter_plant, inter_row, skew, convergence):
        self.offset = offset
        self.position = position
        self.ir_at_bottom = inter_row
        self.ip_at_bottom = inter_plant
        self.convergence = convergence
        self.skew = skew

        self.world = world
        if not (self.world.are_coordinates_valid(self.offset, self.position)):
            print("Warning: particle's offset and/or position has "
                  "an invalid value : ({}, {}).".format(int(self.offset), int(self.position)))

    def get_bottom_plants(self):
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
        leftOffset = self.offset - self.ir_at_bottom
        while self.world.are_coordinates_valid(leftOffset, self.position):
            center = np.asarray([leftOffset, self.position])
            horizontal_neighbors.append(center)
            nb_left_neighbours += 1
            leftOffset -= self.ir_at_bottom

        # Appending the particular plant
        center = np.asarray([self.offset, self.position])
        horizontal_neighbors.append(center)

        # Appending right neighbors of the particular plant
        rightOffset = self.offset + self.ir_at_bottom
        while self.world.are_coordinates_valid(rightOffset, self.position):
            center = np.asarray([rightOffset, self.position])
            horizontal_neighbors.append(center)
            nb_right_neighbours += 1
            rightOffset += self.ir_at_bottom

        return horizontal_neighbors, nb_left_neighbours, nb_right_neighbours

    def get_particular_plant_top_crossing_point(self):
        """
        Returns the coordinates of the crossing point between the row where the particular plant is and the top
        of the image.
        """
        # Use of the tan = opposé / adjacent formula
        x_coordinate = np.tan(self.skew) * self.position
        x_coordinate += self.offset
        center = np.asarray([x_coordinate, 0])

        # The returned position doesn't need to be within the image.
        return center

    def get_all_top_crossing_points(self, nb_left_neighbors, nb_right_neighbors):
        """
        Returns the coordinates of the positions where the rows cross with the top of the image by using the
        convergence.
        """
        # List of coordinates to be returned
        horizontal_neighbors = []

        # Number of plants that have been treated (while loop variable)
        nb_plants_treated = 0

        # Inter-row at the top of the image
        ir_at_top = int(self.convergence * self.ir_at_bottom)

        # Getting the crossing point between the row where the particular plant is and the top of the image.
        top_particular_crossing_point = self.get_particular_plant_top_crossing_point()
        top_offset = top_particular_crossing_point[0]
        top_position = top_particular_crossing_point[1]

        # Appending the left plants
        left_plant_x = top_offset
        while nb_plants_treated < nb_left_neighbors:
            left_plant_x -= ir_at_top
            center = np.asarray([left_plant_x, top_position])
            horizontal_neighbors.append(center)
            nb_plants_treated += 1

        # Appending the top particular crossing point
        center = np.asarray([top_offset, top_position])
        horizontal_neighbors.append(center)
        nb_plants_treated += 1

        # Appending the right plants
        right_plant_x = top_offset
        while nb_plants_treated < (nb_left_neighbors + nb_right_neighbors + 1):
            right_plant_x += ir_at_top
            center = np.asarray([int(right_plant_x), int(top_position)])
            horizontal_neighbors.append(center)
            nb_plants_treated += 1

        return horizontal_neighbors

    def get_vanishing_point(self, bottom_plants, top_crossing_points):
        """
        Takes as input the coordinates of the plants located at the bottom of the image and their associated top
        crossing points.
        Returns the vanishing point : the point where rows are converging in the image perspective.
        """
        # The index of the plants used to get the vanishing point.
        i = 0

        # We loop until we find a valid vanishing point or until we have tried to find the vanishing point using every
        # adjacent rows.
        while i < len(top_crossing_points) and i < len(bottom_plants):

            # Points referring to a first row
            bottom_plant1 = bottom_plants[i]
            top_crossing_point1 = top_crossing_points[i]

            # Points referring to a second row
            bottom_plant2 = bottom_plants[i+1]
            top_crossing_point2 = top_crossing_points[i+1]

            # Finding the coefficient a1 and b1 of the first row line equation a1*x + b1 = y
            a1 = (bottom_plant1[1] - top_crossing_point1[1]) / (bottom_plant1[0] - top_crossing_point1[0])
            b1 = bottom_plant1[1] - a1 * bottom_plant1[0]

            # Finding the coefficient a2 and b2 of the second row line equation a2*x + b2 = y
            a2 = (bottom_plant2[1] - top_crossing_point2[1]) / (bottom_plant2[0] - top_crossing_point2[0])
            b2 = bottom_plant2[1] - a2 * bottom_plant2[0]

            # Finding the coordinates of the vanishing point
            if a1 - a2 == 0:
                vanishing_point_x = (b2 - b1)
            else:
                vanishing_point_x = (b2 - b1) / (a1 - a2)
            vanishing_point_y = a1 * vanishing_point_x + b1

            # Checking if the coordinates we got are not infinite.
            if (not np.isinf(vanishing_point_x)) and (not np.isinf(vanishing_point_y)):
                vanishing_point = np.asarray([vanishing_point_x, vanishing_point_y])
                return vanishing_point

        # If we couldn't find the vanishing point.
        print("Error: couldn't find the vanishing point.")
        return False, (-1, -1)

    def get_inter_plant_distance(self, y, vanishing_point):
        """
        As, in the image perspective, the rows are crossing and not parallel, the inter-plant distance is not constant
        on the y axe.
        Takes a vertical position and the coordinates of the vanishing point.
        Returns the inter-plant distance at this position.
        """
        ip = self.ip_at_bottom * (vanishing_point[1] - y) / (vanishing_point[1] - self.world.height)

        return ip

    def get_row_plants(self, bottom_plant, vanishing_point):
        """
        Returns the coordinates of all the plants located in a row and that are within the image. To do so,
        the algorithm starts at the bottom plant and adds a plant on the line using the inter-plant distance,
        then starts again using this plant as starting point. It ends when we are at the end of the image.
        """
        # List of plants located in the row
        row_plants = []

        # Distance between the bottom plant of the row and the vanishing point
        d = np.sqrt(np.square(vanishing_point[0] - bottom_plant[0])
                    + np.square(vanishing_point[1] - bottom_plant[1]))

        # Ratio between the inter-plant distance at the bottom plant and the total distance d
        ip = self.get_inter_plant_distance(bottom_plant[1], vanishing_point)
        t = ip / d

        # Coordinates of the current plant (in other word while loop variable)
        current_plant = bottom_plant

        # If the inter-plant becomes too small then it means we are very close to the top of the image, so we define the
        # minimal inter-plant distance at which we draw the plants.
        min_ip = 4

        # If 0 < t or t > 1 then it means that the next plant we want to add is outside the image.
        while 0 <= t <= 1 and ip >= min_ip:
            # Coordinates of the next plant on the row
            next_plant_x = (1 - t) * current_plant[0] + t * vanishing_point[0]
            next_plant_y = (1 - t) * current_plant[1] + t * vanishing_point[1]
            next_plant = np.asarray([int(next_plant_x), int(next_plant_y)])

            # Appending the next plant
            if self.world.are_coordinates_valid(next_plant[0], next_plant[1]):
                row_plants.append(next_plant)

            current_plant = next_plant

            # Computing the new value of t
            # print("Current plant : {}".format(current_plant))
            # print("Vanishing point : {}".format(vanishing_point))
            # print("t : {}".format(t))
            d = np.sqrt(np.square(vanishing_point[0] - current_plant[0])
                        + np.square(vanishing_point[1] - current_plant[1]))

            # The new inter-plant distance
            ip = self.get_inter_plant_distance(current_plant[1], vanishing_point)
            t = ip / d

        return row_plants

    def get_row_plants2(self, bottom_plant, vanishing_point):
        """
        Other way of finding all the plants of a row, but it doesn't work as it.
        """
        # List of plants located in the row
        row_plants = []

        # Finding the coefficient a and b of the row line equation a*x + b = y
        a = (bottom_plant[1] - vanishing_point[1]) / (bottom_plant[0] - vanishing_point[0])
        # b = bottom_plant[1] - a * bottom_plant[0]

        # Normalized direction vector
        direction_vector = (1 / np.sqrt(1 + np.square(a)), a / np.sqrt(1 + np.square(a)))

        # Finding the coordinates of all the plants in the row starting from the bottom plant
        current_plant = bottom_plant

        # Scaled direction vector
        ip = self.get_inter_plant_distance(current_plant[1], vanishing_point)
        direction_vector = (direction_vector[0] * ip, direction_vector[1] * ip)

        while self.world.are_coordinates_valid(current_plant[0], current_plant[1]):
            # Appending the current plant
            row_plants.append(current_plant)

            # Coordinates of the possible next plant
            next_plant_x = current_plant[0] + direction_vector[0]
            next_plant_y = current_plant[1] + direction_vector[1]

            current_plant = np.asarray([int(next_plant_x), int(next_plant_y)])

        return row_plants

    def get_all_plants(self):
        """
        Returns a list containing the coordinates of all the plants that the image created by the particle0 contains
        regarding its field parameters.
        """
        # List of all plants coordinates
        plants = []

        # Getting bottom plants coordinates and the number of right and left plants
        bottom_plants, nb_left_plants, nb_right_plants = self.get_bottom_plants()

        # Appending bottom plants
        plants.extend(bottom_plants)

        # Getting the crossing point for each row : point of intersection between a row and the top of the image.
        top_crossing_points = self.get_all_top_crossing_points(nb_left_plants, nb_right_plants)

        # Getting the vanishing point
        # We could take any two plants to compute the vanishing point, here plant 0 and 1
        if len(bottom_plants) < 2 or len(top_crossing_points) < 2:
            print("Can not compute the vanishing point, bottom plants : {}, top crossing points : {}."
                  .format(bottom_plants, top_crossing_points))
            return plants

        vanishing_point = self.get_vanishing_point(bottom_plants, top_crossing_points)

        # Getting all the remaining plants row by row.
        # For each row
        for i in range(len(bottom_plants)):
            bottom_plant = bottom_plants[i]

            # Getting the coordinates of the plants located in the row
            row_plants = self.get_row_plants(bottom_plant, vanishing_point)

            # Appending the plants located in the row
            plants.extend(row_plants)

        return bottom_plants
        return plants

    def get_all_plants_2(self):
        """
        Returns a list containing the coordinates of all the plants that the image created by the particle0 contains
        regarding its field parameters.
        """
        # List of all plants coordinates
        plants = []

        # Getting bottom plants coordinates and the number of right and left plants
        bottom_plants, nb_left_plants, nb_right_plants = self.get_bottom_plants()

        # Appending bottom plants
        plants.extend(bottom_plants)

        # Getting the crossing point for each row : point of intersection between a row and the top of the image.
        top_crossing_points = self.get_all_top_crossing_points(nb_left_plants, nb_right_plants)

        # Getting the vanishing point
        # We could take any two plants to compute the vanishing point, here plant 0 and 1
        if len(bottom_plants) < 2 or len(top_crossing_points) < 2:
            print("Can not compute the vanishing point, bottom plants : {}, top crossing points : {}."
                  .format(bottom_plants, top_crossing_points))
            return plants

        vanishing_point = self.get_vanishing_point(bottom_plants, top_crossing_points)

        # Getting all the remaining plants row by row.
        # For each row
        for i in range(len(bottom_plants)):
            bottom_plant = bottom_plants[i]

            # Getting the coordinates of the plants located in the row
            row_plants = self.get_row_plants(bottom_plant, vanishing_point)

            # Appending the plants located in the row
            plants.extend(row_plants)

        return plants