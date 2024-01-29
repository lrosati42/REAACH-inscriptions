import numpy as np
from scipy.stats import gaussian_kde as KDE
from scipy.signal import find_peaks
from utils import binarize

class treshold_between_peaks:
    def __init__(self, z):
        self.name = "Get the treshold value for binarization purposes through the estimate of the bimodal distribution in the data."
        self.z = z

    def __str__(self):
        return f"{self.name}"

    def get_pdf(self, num_points = 1000):
        self.kernel = KDE(self.z)
        # Create a sample of z values
        self.z_vals = np.linspace(min(self.z), max(self.z), num_points)
        # Compute probability density function on z_vals
        self.pdf_vals = self.kernel(self.z_vals)
    
    def get_peaks(self, div_width = 50, div_distance = 8):
        self.get_pdf()
        num_points = self.z_vals.size
        # Compute second derivative of pdf value
        second_derivative = np.gradient(np.gradient(self.pdf_vals, self.z_vals), self.z_vals)
        # Find local minima with scipy find_peaks
        peaks, _ = find_peaks(-second_derivative, width = int(num_points/div_width), distance = int(num_points/div_distance))
        peak_values = self.kernel(self.z_vals[peaks])
        first = np.argmax(peak_values)
        abs_first = np.argmax(self.pdf_vals)
        peak_values_2nd = np.delete(peak_values, first)
        second = np.where(peak_values == peak_values_2nd.max())[0].tolist()[0]
        self.p1 = self.z_vals[abs_first]
        self.p2 = self.z_vals[peaks[second]]
    
    def get_masked(self):
        self.get_peaks()
        t1 = np.mean(self.z)
        t2 = np.mean([self.p1, self.p2])
        if self.p2 < self.p1:
            treshold = np.mean([t1, t2, self.p2])
        else:
            treshold = t1
        return binarize(self.z, treshold, self.p1, self.p2)
