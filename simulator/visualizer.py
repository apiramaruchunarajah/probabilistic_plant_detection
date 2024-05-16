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

    def draw_particles(self, particle_filter):
        for i in range(particle_filter.n_particles):
            # Coordinates of the particle
            center = np.asarray([int(self.world.width/2), int(particle_filter.particles[i][1][0])])

            perspective_coef = center[1] / self.world.height

            # Color of the particle in fonction of the its weight
            if particle_filter.particles[i][0] > 0.70:
                color = (0, 0, 255)
                thickness = 5
            elif particle_filter.particles[i][0] > 0.40:
                color = (0, 255, 0)
                thickness = 4
            else:
                color = (255, 0, 0)
                thickness = 2

            cv.drawMarker(self.img, center, color, markerType=cv.MARKER_DIAMOND,
                          markerSize=int((7*thickness) * perspective_coef), thickness=thickness)

    def draw(self, plants, particle_filter):
        # Empty image
        self.img = np.zeros((self.world.height, self.world.width, 3), np.uint8)

        # Draw plants and particles
        self.draw_plants(plants)
        self.draw_particles(particle_filter)

        # cv.imshow("Crop rows", self.img)
        # cv.waitKey(0)
