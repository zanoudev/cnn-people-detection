import numpy as np
import os
#import sys
# print("\n".join(sys.path))
# print(cv2.__file__)
from config import THRESHOLD, PERSON_LABEL, MASK_COLOR, ALPHA
from PIL import Image, ImageDraw
from compare import get_person_histograms

## CONFIG
# Seuil pour la classification ou la segmentation
THRESHOLD = 0.5
# ID de label pour la classification des personnes
PERSON_LABEL = 1  
# Couleur du masque pour les visualisations (rouge dans ce cas)
MASK_COLOR = np.array([255, 0, 0]) 
# Valeur alpha pour la transparence ou le mélange
ALPHA = 0.4 

# Fonction pour traiter les sorties d'inférence du modèle
## Modifiée pour calculer les bounding box
def process_inference(model_output, image):

    # Extraire les masques, les scores, et les labels de la sortie du modèle
    masks = model_output[0]['masks']
    scores = model_output[0]['scores']
    labels = model_output[0]['labels']
    boxes = model_output[0]['boxes'] 

    bounding_boxes = []

    # Convertir l'image en tableau numpy
    img_np = np.array(image)

    # Parcourir chaque prédiction pour appliquer le seuil et le label
    for i, (mask, score, label, box) in enumerate(zip(masks, scores, labels, boxes)):
    
        # Appliquer le seuil et vérifier si le label correspond à une personne
        if score > THRESHOLD and label == PERSON_LABEL:

            # coordonnées de la bounding box
            ymin, xmin, ymax, xmax = box.tolist()
            bounding_boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            
            # Convertir le masque en tableau numpy et l'appliquer à l'image            
            mask_np = mask[0].mul(255).byte().cpu().numpy() > (THRESHOLD * 255)            
            for c in range(3):
                img_np[:, :, c] = np.where(mask_np, 
                                        (ALPHA * MASK_COLOR[c] + (1 - ALPHA) * img_np[:, :, c]),
                                        img_np[:, :, c])
    
    # Convertir en image à partir d'un tableau numpy et le renvoyer + bounding boxes    
    result_image = Image.fromarray(img_np.astype(np.uint8))      
    return result_image, bounding_boxes

def get_hist_of_bounding_box(image, coord):
    image = np.array(image)
    x_min, y_min, x_max, y_max = coord
    person_region = image[y_min:y_max, x_min:x_max]
    histograms = get_person_histograms(np.array(person_region))
    return histograms

def draw_bounding_box(image, bbox):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline="red")
    return image

def sort_best_100(result_list):
    sorted_results = sorted(result_list, key=lambda x: x[2], reverse=True)[:100]
    return sorted_results

def save_outputs(result_list, output_path_dir):
    for result in result_list:
        image_name, bbox, score, image  = result
        image.save(os.path.join(output_path_dir, image_name))