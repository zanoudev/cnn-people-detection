# Multi-Camera Person Re-Identification

A computer vision project using instance segmentation and RGB color histogram analysis to identify individuals across multiple frames of security footage. The system leverages a pretrained convolutional neural network for person detection and applies color-based similarity metrics to track known individuals over time and across different camera angles.

## Getting Started

**Prerequisites:**

- Python 3.8 or higher  
- Virtual environment (recommended)

## Setup

**Create and activate a virtual environment:**

<details>
<summary>On Windows</summary>

```bash
python -m venv venv
source venv/Scripts/activate
```

</details>

<details>
<summary>On Unix-based systems</summary>

```bash
python -m venv venv
source venv/bin/activate
```

</details>

**Install the required dependencies:**

```bash
pip install -r requirements.txt
```

> Ensure that `torch` and `torchvision` are installed according to your system configuration:  
> https://pytorch.org/get-started/locally/

---

## Part 1: Histogram-Based Matching (Baseline Method)

This component implements a basic person matching approach using manually defined bounding boxes and HSV histogram comparison. The method compares individuals between a set of test images and video frames using their color distribution.

### Methodology

- Each individual in the test images is identified via bounding boxes.
- For each bounding box, both full-body and upper-body regions are extracted.
- Normalized HSV histograms are computed for these regions.
- These histograms are compared against every person detected in a sequence of video frames.
- A similarity score is calculated using histogram intersection.
- The top 100 matches are retained and exported for evaluation.

### Input Requirements

- A folder containing image frames from a video sequence.
- A `labels.txt` file with bounding box coordinates for individuals in the video frames.
- Two test images containing reference individuals.

---

## Part 2: Mask R-CNN and RGB Histogram Matching

This is the primary component of the project. It replaces manual annotations with automated person segmentation and improves matching accuracy through the use of RGB histograms and a CNN-based detector.

### Methodology

- A pretrained Mask R-CNN model is used to detect individuals in each frame.
- Five reference individuals are provided via a set of static images.
- For each reference image, RGB histograms are extracted from the full body and upper body.
- In every frame, detected individuals are segmented, and their RGB histograms are computed in the same way.
- Histogram intersection is used to compare detected individuals with the references.
- Matches with scores exceeding a configurable threshold are retained.
- For each reference individual, the top 100 matches are saved to disk.

---

## Output

- Each individual’s best matches are saved as image files in separate output directories.
- Results can be verified visually to assess the effectiveness of the matching process.
