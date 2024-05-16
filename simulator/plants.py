import numpy as np
import cv2 as cv


# Weeds needs to be added
class Plants:
    def __init__(self, world, vp_height, vp_width, ir, ip, o, s, c, nb_rows, nb_plant_types):
        # Initialize plants positions
        # World contains the width and height of the image
        self.world = world

        # Parameters to generate image and to be tracked
        self.inter_row_distance = ir
        self.inter_plant_distance = ip
        self.offset = o  # between -(width/2) and +(width/2)
        self.skew = s  # TODO
        self.convergence = c  # TODO

        # Field parameters
        self.vp_height = vp_height
        self.vp_width = vp_width + self.offset
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

    def generate_plants(self):
        for i in range(self.nb_rows):
            # Empty image
            img = np.zeros((self.world.height, self.world.width), np.uint8)

            # Get position of lines crossing in the vanishing point
            cv.line(img, (i * self.inter_row_distance + self.offset, self.world.height), self.vanishing_point, 255, 1)
            coordinates = np.where(img != 0)
            row_coordinates = np.array(list(zip(coordinates[1], coordinates[0])))
            self.plants_rows.append(row_coordinates)

            # Generate initial positions of the plants for a row
            # number of plants in the row : between 70% and 100% of the maximum number of plants per row
            nb_plants = np.random.randint(np.floor(0.70 * self.max_number_plants_per_row),
                                          self.max_number_plants_per_row)
            # random positions for each plant
            random_selection = np.random.choice(np.arange(self.vp_height, self.world.height, self.inter_plant_distance),
                                                nb_plants, replace=False)
            # add of a little noise
            for j in range(len(random_selection)):
                selection = random_selection[j]
                # Gaussian centered around the original coordinates
                random_selection[j] = np.random.normal(selection, 11, 1)

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
