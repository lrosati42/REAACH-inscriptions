import numpy as np
import pywavefront
from scipy.stats import zscore
import matplotlib.pyplot as plt

def load_obj(path: str, filename: str):
        obj = pywavefront.Wavefront(f"{path}/{filename}.obj")
        return np.array(obj.vertices).T[:3].T

def exclude_outliers(z, threshold = 3):
        zscores = np.abs(zscore(z))
        keep_idx = (zscores < threshold).nonzero()[0]
        return keep_idx

def reduce_data(x, y, z):
        keep_idx = exclude_outliers(z, threshold = 3)
        x = x[keep_idx]
        y = y[keep_idx]
        z = z[keep_idx]
        return x, y, z

def binarize(z, threshold, p1, p2):
        # return a (binarized) mask of points above (white) and below (black) the threshold
        return z < threshold

def rotation_matrix(axis, theta):
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
def rotate_array_of_points(array_of_points, axis_of_rotation, angle_of_rotation):
        R = rotation_matrix(axis_of_rotation, angle_of_rotation)
        rotated_array = np.dot(R, array_of_points.T).T
        return rotated_array

def check_sign(points, ref):
        z = points.T[-1].T
        if np.median(z) < np.mean(z):
                # cloud point is upside down, so flip it
                ref = ref / np.linalg.norm(ref)
                u_r = -ref
                v = ref
                # flip over x axis
                axis_of_rotation = np.array([1,0,0])
                angle_of_rotation = np.arccos(np.dot(u_r, v))
                rotated = rotate_array_of_points(points, axis_of_rotation, angle_of_rotation)
        else:
                # no need to flip
                rotated = points

        x = rotated.T[0].T
        y = rotated.T[1].T
        z = rotated.T[2].T

        return x, y, z

def save_processed_image(x, y, z, path, filename):
        fig = plt.figure()
        # Point cloud visualization
        plt.scatter(x, y, c=z, cmap='Greys', marker='.')
        ax = fig.gca()
        # remove axis, margins and borders from the visualization
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.set_aspect('equal')

        # save the figure
        plt.savefig(f"{path}/{filename}.png", bbox_inches='tight', pad_inches=0)