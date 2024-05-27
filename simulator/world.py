class World:
    def __init__(self, width, height, speed):
        self.width = width
        self.height = height
        self.speed = speed

    def are_coordinates_valid(self, x, y):
        """
        Returns a boolean indicating whether (x, y) is inside the world/image or not (valid position or not).
        """
        return not (x < 0 or x >= self.width or y < 0 or y >= self.height)
