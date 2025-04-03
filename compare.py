import cv2
import os
import numpy as np
from PIL import Image

def calc_histogram_rgb(image):
    """Calculates RGB histogram for an image."""
    #r_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    histogram = cv2.calcHist([np.array(image)], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    histogram /= histogram.sum()
    return histogram


def get_upper_half(image):
    """Gets the upper half of an image."""
    if isinstance(image, np.ndarray):
        # Convert NumPy array to PIL image
        image = Image.fromarray(image)

    width, height = image.size
    upper_half = image.crop((0, 0, width, height // 2))
    return upper_half


def compare_histograms(hist1, hist2):
    """Compares histograms using intersection."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

def compare_score(request_pHist, detected_pHist):
    """Determines if 2 histograms are similar enough"""
    a = compare_histograms(request_pHist[0], detected_pHist[0])
    b = compare_histograms(request_pHist[1], detected_pHist[1])
    c = compare_histograms(request_pHist[0], detected_pHist[1])
    d = compare_histograms(request_pHist[1], detected_pHist[0])

    return max(a,b,c,d)

def get_person_histograms(person_region):
    histograms = [] # une liste de liste, contient les histogrammes des personnes, en ordre d'apparition dans labels

    person_hist = calc_histogram_rgb(person_region)
    upper_half = get_upper_half(person_region)
    upper_hist = calc_histogram_rgb(upper_half)

    histograms.append(person_hist)
    histograms.append(upper_hist)
    return histograms