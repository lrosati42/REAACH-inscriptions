from argparse import ArgumentParser
import utils
from orient import rotate
from treshold import treshold_between_peaks
import numpy as np

parser = ArgumentParser()
parser.add_argument("-d", "--directory", dest="in_path",
                    help="directory containing the obj files")
parser.add_argument("-f", "--file", dest="filename",
                    help="name of the obj file to process")
parser.add_argument("-o", "--output", dest="out_path",
                    help="directory to save processed image")
args = parser.parse_args()

# get chosen path and filename
in_path = args.in_path
filename = args.filename
out_path = args.out_path

# get points
points = utils.load_obj(path = in_path, filename = filename)
print("Point cloud loaded.")

# Rotate data in order to match target orientation
ref_normal = np.array([0, 0, 1])
rotation = rotate(points = points, reference = ref_normal)
x, y, z = rotation.get_rotated_coordinates()
print("Point cloud rotated.")

# Discard outliers in the z-coordinate
x, y, z = utils.reduce_data(x, y, z)
print("Point cloud reduced.")

# Check if the points are upside down
points = np.stack([x,y,z]).T
x, y, z = utils.check_sign(points, ref_normal)

# Binarize data (resulting into a b/w image)
binarizing = treshold_between_peaks(z)
z_mask = binarizing.get_masked()
print("Point cloud binarized.")

# Save processed image
utils.save_processed_image(x, y, z_mask, out_path, filename)
print(f"Results saved in {out_path}{filename}.png.")