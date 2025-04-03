import cv2
import os
import numpy as np


    
test_image1 = "1636738357390407600"
test_image2 = "1636738315284889400"

# Load test images
test_images = [cv2.imread(test_image1 + ".png"), cv2.imread(test_image2 + ".png")]

persons_ids = []

def calc_histogram_rgb(image):
    """Calculates RGB histogram for an image."""
    histogram = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    histogram /= histogram.sum()
    return histogram


def get_upper_half(image):
    """Gets the upper half of an image."""
    width, height, _ = image.shape
    upper_half = image[0:height // 2, :]
    return upper_half

def get_image_from_line(line):
    data = line.strip().split(',')
    image = data[0]
    return image + '.png'

def get_line_from_image(image_file):
    image_name = image_file.split('.')[0]




def get_histograms(image_name):
    histograms = [] # une liste de liste, contient les histogrammes des personnes, en ordre d'apparition dans labels
    j=0
    for line in lines:
        #j+=1
        #print(str(j) + " + ligne " + line )
        if image_name in line:
            data = line.strip().split(',')
            position = data[1:]
            x = int(position[0])
            y = int(position[1])
            w = int(position[2])
            h = int(position[3])

            if w*h < 2500 :
                continue #skip to next line
            
            image_file = image_name+'.png'
            image = cv2.imread(os.path.join(folder,image_file))

            if image is None: continue # handling non existing images

            person_region = image[y:y + h, x:x + w]


            person_hist = calc_histogram_rgb(person_region)
            upper_half = get_upper_half(person_region)
            upper_hist = calc_histogram_rgb(upper_half)

            person_histograms = [] # position 0 contient full_hist et position 1 contient  

            person_histograms.append(person_hist)
            person_histograms.append(upper_hist)
            histograms.append(person_histograms)
    return histograms



def compare_histograms(hist1, hist2):
    """Compares histograms using intersection."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)


def find_similar_images(person_hist): #[hist, hist]
    maxV = []
    j=0
    for line in lines:
        print(j)
        j+=1
        data = line.strip().split(',')
        image_name = data[0]
        image_hist = get_histograms(image_name) # [p=[hist, hist],p=[hist,hist],[hist, hist]] 
        for p in image_hist:
            a = compare_histograms(p[0], person_hist[0])
            b = compare_histograms(p[1], person_hist[1])
            c = compare_histograms(p[0], person_hist[1])
            d = compare_histograms(p[1], person_hist[0])
            maxV.append((max(a,b), image_name))
    maxV = sorted(maxV, key=lambda a: a[0], reverse=True) # check for sorted list of tuples
    maxV = maxV[:100]
    return maxV



