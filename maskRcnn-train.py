import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo  # Correct import for model_zoo
import os
import random
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer

# # Step 1: Register your custom dataset in COCO format
# def register_custom_dataset():
#     dataset_name = "room-seg-v2-final.v1i.coco-segmentation"

#     # Check if the dataset is already registered to avoid re-registering it
#     if dataset_name not in DatasetCatalog.list():
#         # Path to images and annotations
#         image_dir = "H:/wall-segmenataion/room-seg-v2-final.v1i.coco-segmentation/train/"
#         json_annotation = "H:/wall-segmenataion/room-seg-v2-final.v1i.coco-segmentation/train/_annotations.coco.json"

#         # Register the dataset
#         register_coco_instances(dataset_name, {}, json_annotation, image_dir)

#         # Set the class name (Assuming we only need one class "Room")
#         MetadataCatalog.get(dataset_name).thing_classes = ["none","room"]
#     else:
#         print(f"Dataset '{dataset_name}' is already registered.")

# # Call the function to register the dataset
# register_custom_dataset()
# Step 1: Register your custom dataset in COCO format
def register_custom_dataset():
    dataset_name = "room-seg-v2-final.v1i.coco-segmentation"

    # Check if the dataset is already registered
    if dataset_name in DatasetCatalog.list():
        print(f"Dataset '{dataset_name}' is already registered, unregistering it first...")
        # Unregister the dataset
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)

    # Path to images and annotations
    image_dir = "H:/wall-segmenataion/room-seg-v2-final.v1i.coco-segmentation/train/"
    json_annotation = "H:/wall-segmenataion/room-seg-v2-final.v1i.coco-segmentation/train/_annotations.coco.json"

    # Register the dataset
    register_coco_instances(dataset_name, {}, json_annotation, image_dir)

    # Set the class name (Assuming we only need one class "Room")
    MetadataCatalog.get(dataset_name).thing_classes = ["none","room"]

    print(f"Dataset '{dataset_name}' registered successfully.")

# Call the function to register the dataset
register_custom_dataset()

# Step 2: Visualize some samples (updated to use matplotlib for Colab)
def visualize_samples():
    dataset_dicts = DatasetCatalog.get("room-seg-v2-final.v1i.coco-segmentation")
    metadata = MetadataCatalog.get("room-seg-v2-final.v1i.coco-segmentation")

    for d in random.sample(dataset_dicts, 3):  # Visualize 3 random samples
        img = cv2.imread(d["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display in matplotlib
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)

        # Display the image using matplotlib in Colab
        plt.figure(figsize=(12, 8))
        plt.imshow(vis.get_image()[:, :, ::-1])  # Convert from BGR to RGB
        plt.axis("off")
        plt.show()

# Uncomment the line below to visualize the data samples in Colab
visualize_samples()

# Step 3: Set up the Mask R-CNN fine-tuning config and run training
def train_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Set dataset and output paths
    cfg.DATASETS.TRAIN = ("room-seg-v2-final.v1i.coco-segmentation",)
    cfg.DATASETS.TEST = ()  # No validation dataset in this case
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = "H:/wall-segmenataion/"  # Path to save model weights and outputs

    # Use the pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Set the number of classes (1 class: "Room")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # Set training parameters
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust based on your GPU memory
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = 1000  # Number of iterations (adjust based on your dataset size)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Number of proposals per image

    # Create the output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Initialize the trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Start training
    trainer.train()

# Run the training function to fine-tune the model
train_model()
