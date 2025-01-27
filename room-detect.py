import cv2
import torch
import json
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from PIL import Image
import fitz

# Step 1: Set up the inference configuration and initialize the model
def setup_inference():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Path to trained weights
    cfg.MODEL.WEIGHTS = "models/models-room-detection-weights/model_final.pth"  # Update to your model weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the confidence threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Number of classes ("none" and "room")

    cfg.DATASETS.TEST = ("room-seg-v2-final.v1i.coco-segmentation",)

    # Check if GPU is available, else use CPU
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return DefaultPredictor(cfg)

# Step 2: Perform inference and return results
def run_inference(predictor, image_path, output_folder):
    # Load the image
    img = cv2.imread(image_path)
    outputs = predictor(img)

    # Extract instances
    instances = outputs["instances"].to(torch.device("cpu"))
    bboxes = instances.pred_boxes.tensor.numpy().tolist()
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy().tolist()
    classes = instances.pred_classes.numpy().tolist()

    # Prepare JSON output
    results = []
    for i in range(len(bboxes)):
        results.append({
            "bbox": bboxes[i],
            "mask": masks[i].tolist(),  # Convert numpy array to list
            "score": scores[i],
            "class": classes[i]
        })

    # Save the JSON file
    base_name = os.path.basename(image_path)
    json_output_path = os.path.join(output_folder, os.path.splitext(base_name)[0] + "_output.json")
    with open(json_output_path, "w") as f:
        json.dump(results, f, indent=4)

    # Return the results and JSON path
    return results, json_output_path

# Step 3: Visualize the image with detections
def visualize_inference(image_path, predictor, output_folder):
    # Load the image
    img = cv2.imread(image_path)
    metadata = MetadataCatalog.get("room-seg-v2-final.v1i.coco-segmentation")

    # Perform inference
    outputs = predictor(img)

    # Visualize the results
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
    vis = visualizer.draw_instance_predictions(outputs["instances"].to(torch.device("cpu")))

    # Save the visualized image
    base_name = os.path.basename(image_path)
    visualized_image_path = os.path.join(output_folder, os.path.splitext(base_name)[0] + "_visualized.jpg")
    cv2.imwrite(visualized_image_path, vis.get_image()[:, :, ::-1])

    return visualized_image_path

# Step 4: Convert PDF to images using PyMuPDF
def pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)  # Open the PDF document
     # Create a scaling matrix
    for page_number in range(len(doc)):
        page = doc[page_number]
        pix = page.get_pixmap() # Apply the scaling matrix
        
        # Save the image for the current page
        image_path = os.path.join(
            output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_number + 1}.jpg"
        )
        pix.save(image_path)
        yield image_path

# Step 5: Process all images in a folder
def process_folder(input_folder, output_folder, predictor):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {file_name}")
            run_inference(predictor, file_path, output_folder)
            visualize_inference(file_path, predictor, output_folder)

        elif file_name.lower().endswith('.pdf'):
            print(f"Processing PDF: {file_name}")
            for image_path in pdf_to_images(file_path, output_folder):
                run_inference(predictor, image_path, output_folder)
                visualize_inference(image_path, predictor, output_folder)

# Step 6: Main function for inference
def main():
    input_folder = "input"  # Folder containing images and PDFs
    output_folder = "output-inference"  # Folder to save outputs
    predictor = setup_inference()

    process_folder(input_folder, output_folder, predictor)

if __name__ == "__main__":
    main()