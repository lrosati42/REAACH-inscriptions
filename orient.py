import numpy as np
from skspatial.objects import Plane
from utils import rotate_array_of_points

class rotate:
    def __init__(self, points, reference):
        self.name = "Point cloud rotation."
        self.points = points
        self.reference = reference

    def __str__(self):
        return f"{self.name}"

    def get_points_normal(self, num_points = 5000):
        # downsample to a reasonable number of points for the plane fit
        keep_idx = np.random.choice(self.points.shape[0], num_points)
        plane = Plane.best_fit(self.points[keep_idx])
        self.normal = plane.normal

    def get_reference_normal(self):
        # for now the reference vector is passed as an argument
        return self.reference
    
    def align(self):
        self.get_points_normal()
        u = self.normal
        v = self.reference
        axis_of_rotation = np.cross(u, v)
        angle_of_rotation = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        rotated = rotate_array_of_points(self.points, axis_of_rotation, angle_of_rotation)
        return rotated

    def get_rotated_coordinates(self):
        rotated_points = self.align()
        x = rotated_points.T[0]
        y = rotated_points.T[1]
        z = rotated_points.T[2]
        return x, y, z