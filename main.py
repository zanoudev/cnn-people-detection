import os
from PIL import Image
#from utils import model, tools, compare
import model, tools, compare
import torch
import numpy as np

# Point d'entrée principal du script
if __name__ == "__main__":

    # Définir les répertoires source et de sortie, et le nom de l'image
    test_dir = "images\\test"
    #images = [test_dir]

    cam0_dir = "images\cam0"
    cam1_dir = "images\cam1"
    #images = [cam0_dir, cam1_dir]
    images = [cam0_dir]
    

    persons_img_dir = "inst_seg\persons_img"
    persons_images = os.listdir(persons_img_dir) # images of persons inside the directory

    # Charger le modèle et appliquer les transformations à l'image
    seg_model, transforms = model.get_model()
    # comparaison threshold
    threshold = 0.05

    person_num = 1
    request_persons_histograms = []
    results_dict = {}
    results_dict[1] = []
    results_dict[2] = []
    results_dict[3] = []
    results_dict[4] = []
    results_dict[5] = []

    for person in persons_images:  # will iterate 5 times
        print("Calculating histogram of person " + str(person_num))

        person_img_path = os.path.join(persons_img_dir, person)
        person_img = Image.open(person_img_path)

        # getting the histograms of request person's image
        request_person_hist = compare.get_person_histograms(person_img)
        request_persons_histograms.append([person_num, request_person_hist])  
        person_num += 1
        image_count=0

    for cam in images: #cam0 et cam1
        sequence = os.listdir(cam)
        for image_name in sequence:
            image_count += 1
            print("image " + str(image_count))

            image_path = os.path.join(cam, image_name)
            image = Image.open(image_path)
            transformed_img = transforms(image)

            # Effectuer l'inférence sur l'image transformée sans calculer les gradients
            with torch.no_grad():
                output = seg_model([transformed_img])
                # Traiter le résultat de l'inférence
                result_image, b_boxes = tools.process_inference(output,image)
                print(b_boxes)

                # will iterate n times (n is the number of persons detected)
                for bbox in b_boxes:
                    detected_p_histograms = tools.get_hist_of_bounding_box(np.array(image), bbox)
                    for request_person in request_persons_histograms:
                        person_id, request_person_hist = request_person
                        score = compare.compare_score(detected_p_histograms, request_person_hist)
                        if ( score > threshold):
                            #results_dict[person_id] = []
                            results_dict[person_id].append([image_name,bbox,score, image])
                            break
                            # draw bounding box in a copy of the image
                            #result = tools.draw_bounding_box(result_image, bbox)
                        else: continue
                #end for
            #end with
        #end for
    #end for
    
    person1 = results_dict[1]
    person2 = results_dict[2]
    person3 = results_dict[3]
    person4 = results_dict[4]
    person5 = results_dict[5]

    person1 = tools.sort_best_100(person1)
    tools.save_outputs(person1, "inst_seg\outputs\output_p1")

    person2 = tools.sort_best_100(person2)
    tools.save_outputs(person2, "inst_seg\outputs\output_p2")

    person3 = tools.sort_best_100(person3)
    tools.save_outputs(person3, "inst_seg\outputs\output_p3")

    person4 = tools.sort_best_100(person4)
    tools.save_outputs(person4, "inst_seg\outputs\output_p4")

    person5 = tools.sort_best_100(person5)
    tools.save_outputs(person5, "inst_seg\outputs\output_p5")


