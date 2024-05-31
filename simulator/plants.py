import numpy as np
import cv2 as cv


# Weeds needs to be added
class Plants:
    def __init__(self, world, vp_height, vp_width, ir, ip, o, nb_rows, nb_plant_types):
        # Initialize plants positions
        # World contains the width and height of the image
        self.world = world

        # Parameters to generate image and to be tracked
        # As we directly give the coordinates of the vanishing point, we don't need the parameters skew or convergence.
        self.inter_row_distance = ir
        self.inter_plant_distance = ip
        self.offsett = o
        # Offset is between -(width/2) and +(width/2), it doesn't correspond to the definition of offsett used to create
        # a particle (which in that case goes from 0 to width and just corresponds to the x coordinate of the particular
        # plant).

        # Field parameters
        self.vp_height = vp_height
        self.vp_width = vp_width + self.offsett
        self.vanishing_point = (vp_width, vp_height)

        # Plants generation parameters
        vp_distance = np.absolute(world.height - vp_height)

        self.nb_rows = nb_rows
        self.nb_plant_types = nb_plant_types
        self.max_number_plants_per_row = np.ceil(vp_distance / self.inter_plant_distance)

        # List containing positions of all the plants
        self.plant_positions = []
        self.plants_rows = []
        self.plant_types = []

        # Initialize standard deviation noise for plants motion
        self.std_move_distance = 0

        # Initialize standard deviation noise for plants position measurement
        self.std_meas_position = 0

    def setStandardDeviations(self, std_move_distance, std_meas_position):
        self.std_move_distance = std_move_distance
        self.std_meas_position = std_meas_position

    def get_inter_plant_distance(self, y):
        """
        As, in the image perspective, the rows are crossing and not parallel, the inter-plant distance is not constant
        on the y axe.
        Takes a vertical position and the coordinates of the vanishing point.
        Returns the inter-plant distance at this position.
        """
        ip = self.inter_plant_distance * (self.vanishing_point[1] - y) / (self.vanishing_point[1] - self.world.height)

        return ip

    def get_row_plants(self, bottom_plant):
        """
        Returns the coordinates of all the plants located in a row and that are within the image. To do so,
        the algorithm starts at the bottom plant and adds a plant on the line using the inter-plant distance,
        then starts again using this plant as starting point. It ends when we are at the end of the image.
        """
        # List of plants located in the row
        row_plants = []

        # Distance between the bottom plant of the row and the vanishing point
        d = np.sqrt(np.square(self.vanishing_point[0] - bottom_plant[0])
                    + np.square(self.vanishing_point[1] - bottom_plant[1]))

        # Ratio between the inter-plant distance at the bottom plant and the total distance d
        ip = self.get_inter_plant_distance(bottom_plant[1], self.vanishing_point)
        t = ip / d

        # Coordinates of the current plant (in other word while loop variable)
        current_plant = bottom_plant

        # If the inter-plant becomes too small then it means we are very close to the top of the image, so we define the
        # minimal inter-plant distance at which we draw the plants.
        min_ip = 4

        # If 0 < t or t > 1 then it means that the next plant we want to add is outside the image.
        while 0 <= t <= 1 and ip >= min_ip:
            # Coordinates of the next plant on the row
            next_plant_x = (1 - t) * current_plant[0] + t * self.vanishing_point[0]
            next_plant_y = (1 - t) * current_plant[1] + t * self.vanishing_point[1]
            next_plant = np.asarray([int(next_plant_x), int(next_plant_y)])

            # Appending the next plant
            if self.world.are_coordinates_valid(next_plant[0], next_plant[1]):
                row_plants.append(next_plant)

            current_plant = next_plant

            # Computing the new value of t
            # print("Current plant : {}".format(current_plant))
            # print("Vanishing point : {}".format(vanishing_point))
            # print("t : {}".format(t))
            d = np.sqrt(np.square(self.vanishing_point[0] - current_plant[0])
                        + np.square(self.vanishing_point[1] - current_plant[1]))

            # The new inter-plant distance
            ip = self.get_inter_plant_distance(current_plant[1], self.vanishing_point)
            t = ip / d

        return row_plants

    def get_bottom_plants(self, offset, position):
        # List of coordinates to be returned
        horizontal_neighbors = []

        # Number of left and right neighbors
        nb_left_neighbours = 0
        nb_right_neighbours = 0

        # Appending the left neighbors of the particular plant
        leftOffset = self.offset - self.inter_row_distance
        while self.world.are_coordinates_valid(leftOffset, position):
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

    def generate_plants(self):
        for i in range(self.nb_rows):
            # Empty image
            img = np.zeros((self.world.height, self.world.width), np.uint8)

            # Get position of lines crossing in the vanishing point
            cv.line(img, (i * self.inter_row_distance + self.offsett, self.world.height), self.vanishing_point,
                    255, 1)
            coordinates = np.where(img != 0)
            row_coordinates = np.array(list(zip(coordinates[1], coordinates[0])))
            self.plants_rows.append(row_coordinates)

            # Generate initial positions of the plants for a row
            # number of plants in the row : between 70% and 100% of the maximum number of plants per row
            nb_plants = np.random.randint(np.floor(0.70 * self.max_number_plants_per_row),
                                          self.max_number_plants_per_row)
            nb_plants = int(self.max_number_plants_per_row)

            # random positions for each plant
            random_selection = np.random.choice(np.arange(self.vp_height, self.world.height, self.inter_plant_distance),
                                                nb_plants, replace=False)
            # add of a little noise
            for j in range(len(random_selection)):
                selection = random_selection[j]
                # Gaussian centered around the original coordinates
                random_selection[j] = np.random.normal(selection, 11, 1)
                random_selection[j] = selection

            self.plant_positions.append(random_selection)

            # Mapping a type for each plant
            plant_markers = np.random.choice(self.nb_plant_types, nb_plants)
            self.plant_types.append(plant_markers)

        # Find highest plant across all rows
        # self.highest_plant = np.min(sum([list(sublist) for sublist in self.initial_plant_positions], []))

    def move(self, desired_move_distance):
        # Compute relative motion (true motion is desired motion with some noise)
        move_distance = np.random.normal(loc=desired_move_distance, scale=self.std_move_distance, size=1)[0]

        # Move every plants
        for row_idx in range(self.nb_rows):
            for i in range(len(self.plant_positions[row_idx])):
                self.plant_positions[row_idx][i] += move_distance

    def measure(self):
        """
        Returns the height of the tracked plant.
        The tracked plant being the plant of the middle row that is the closest to the bottom of the image
        """

        tracked_plant_height = -1

        for row_idx in range(self.nb_rows):
            # If we are in the middle row
            if row_idx == np.floor(self.nb_rows / 2):
                current_plants_positions = np.asarray(self.plant_positions[row_idx])
                last_point_of_row = self.plants_rows[row_idx][-1][1]
                visible_plants_positions = current_plants_positions[
                    (current_plants_positions >= 0) & (current_plants_positions <= last_point_of_row)]

                try:
                    tracked_plant_height = np.max(visible_plants_positions)
                except:
                    print("The last plant to track has gone")

        # Adding noise for the measurement
        tracked_plant_height_with_noise = np.random.normal(loc=tracked_plant_height,
                                                           scale=self.std_meas_position, size=1)[0]

        return tracked_plant_height_with_noise

    def getPlantsToDraw(self, row_idx):
        """
        Returns a list containing the positions and a list containing the type of the plants to draw given a row index.
        Used by the visualizer.

        :param row_idx: Index of the row where we look for plants
        :returns (list containing the positions, list containing the types):
        """

        if row_idx < 0 or row_idx >= self.nb_rows:
            print("Error the row index is incorrect")
            return -1

        current_plants_positions = np.asarray(self.plant_positions[row_idx])
        last_point_of_row = self.plants_rows[row_idx][-1][1]
        visible_plants_positions = current_plants_positions[
            (current_plants_positions >= 0) & (current_plants_positions <= last_point_of_row)]

        plant_positions = self.plants_rows[row_idx][visible_plants_positions]
        plant_types = self.plant_types[row_idx][
            (current_plants_positions >= 0) & (current_plants_positions <= last_point_of_row)]

        return plant_positions, plant_types
