import os
import cv2
import json
import numpy as np
from fastai.vision.all import *
from pathlib import Path
from PIL import Image
import fitz

# === Prediction and Saving ===
def predict_and_save(image_path, output_folder):
    # Load the image
    img = PILImage.create(str(image_path)).resize((1024, 1024))
    # Perform prediction
    pred, _, _ = learn.predict(img)
    # Resize prediction to match original image size
    if isinstance(pred, torch.Tensor):
        pred = Image.fromarray(pred.numpy().astype(np.uint8))
    resized_pred = pred.resize(img.size, Image.BICUBIC)
    resized_pred_np = np.array(resized_pred)
    # Normalize and convert to uint8 if needed
    if resized_pred_np.max() <= 1:
        resized_pred_np = (resized_pred_np * 255).astype(np.uint8)
    # Save the prediction mask
    mask_output_path = output_folder / f"{Path(image_path).stem}_mask.jpg"
    mask_image = Image.fromarray(resized_pred_np)
    mask_image.save(mask_output_path)
    # Save the resized input image to the output folder
    resized_img_output_path = output_folder / f"{Path(image_path).stem}.jpg"
    img.save(resized_img_output_path)
    print(f"Processed: {image_path}")
    return resized_img_output_path, mask_output_path

# === Overlay Mask ===
def overlay_mask_on_image(original_img_path, mask_img_path, output_path):
    original_img = cv2.imread(original_img_path)
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    if original_img is None or mask_img is None:
        raise FileNotFoundError("Image or mask not found.")

    inverted_mask = cv2.bitwise_not(mask_img)
    normalized_mask = inverted_mask.astype(np.float32) / 255.0
    yellow_color = [0, 0, 255]
    overlayed_img = original_img.copy()
    overlayed_img[inverted_mask == 0] = yellow_color

    cv2.imwrite(output_path, overlayed_img)
    print(f"Overlay image of {original_img_path} saved to: {output_path}")
    return normalized_mask

# === Mask to Polygon ===
def mask_to_polygon(mask_array, json_output_path, image_name, original_image_path, visualized_output_path):
    binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    height, width = mask_array.shape
    for idx, contour in enumerate(contours):
        polygon = contour.reshape(-1, 2)
        polygons.append(polygon.tolist())
        # print(f"Contour {idx+1}: Points = {len(polygon)}")

    polygons = [poly for poly in polygons if len(poly) >= 3]    
    output_data = {
        "image_name": image_name,
        "mask_dimensions": [height, width],
        "polygons": polygons
    }
    with open(json_output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    original_image = cv2.imread(original_image_path)
    for polygon in polygons:
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(original_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(visualized_output_path, original_image)
    print(f"Saved {len(polygons)} polygons to {json_output_path}")

# === PDF to Images ===
def pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)  # Open the PDF document
    # scale = dpi / 72
    # matrix = fitz.Matrix(scale, scale)  # Create a scaling matrix
    image_paths = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)

        image_path = os.path.join(
            output_folder, f"{Path(pdf_path).stem}_page_{page_number + 1}.jpg"
        )
        pix.save(image_path)
        image_paths.append(image_path)
    return image_paths

# === Main Processing ===
def process_folder(input_folder, output_folder):
    for file_path in Path(input_folder).glob("*.*"):
        try:
            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                print(f"Processing image: {file_path}")
                resized_image_path, binary_mask_path = predict_and_save(file_path, output_folder)

                overlay_output_path = os.path.join(output_folder, f"{Path(file_path).stem}_overlay.jpg")
                normalized_mask = overlay_mask_on_image(resized_image_path, binary_mask_path, overlay_output_path)

                json_output_path = os.path.join(output_folder, f"{Path(file_path).stem}_polygons.json")
                visualized_output_path = os.path.join(output_folder, f"{Path(file_path).stem}_polygons.jpg")
                mask_to_polygon(normalized_mask, json_output_path, Path(file_path).name, resized_image_path, visualized_output_path)

            elif file_path.suffix.lower() == ".pdf":
                print(f"Processing PDF: {file_path}")
                image_paths = pdf_to_images(file_path, output_folder)
                for image_path in image_paths:
                    resized_image_path, binary_mask_path = predict_and_save(image_path, output_folder)

                    overlay_output_path = os.path.join(output_folder, f"{Path(image_path).stem}_overlay.jpg")
                    normalized_mask = overlay_mask_on_image(resized_image_path, binary_mask_path, overlay_output_path)

                    json_output_path = os.path.join(output_folder, f"{Path(image_path).stem}_polygons.json")
                    visualized_output_path = os.path.join(output_folder, f"{Path(image_path).stem}_polygons.jpg")
                    mask_to_polygon(normalized_mask, json_output_path, Path(image_path).name, resized_image_path, visualized_output_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    # === Configuration ===
    input_folder = "input-wall"
    output_folder = Path("output-wall-segmentation-2")
    os.makedirs(output_folder, exist_ok=True)

    # === Model Initialization ===
    # Set the path to your data
    path = Path("models/models-unet-wall-seg-weights/masked-data-v3")
    path_lbl = path/'walllabels'
    path_img = path/'images'
    codes = ['wall', 'not-wall']; codes
    name2id = {v:k for k,v in enumerate(codes)}
    wall_code = name2id['wall']


    def acc_wall(input, target):
        target = target.squeeze(1)
        return (input.argmax(dim=1)==target).float().mean()

    def precision(input, target):
        target = target.squeeze(1)
        input = input.argmax(dim=1)
        mask = input == wall_code
        return (input[mask]==target[mask]).float().mean()

    def recall(input, target):
        target = target.squeeze(1)
        mask = target == wall_code
        return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


    fnames = get_image_files(path_img)
    lbl_names = get_image_files(path_lbl)
    lbl_names[:3]
    img_f = fnames[4]


    get_y_fn = lambda x: path_lbl/f'{x.stem}.png'
    get_y_fn(img_f)
    mask = PILMask.create(get_y_fn(img_f))
    src_size = np.array(mask.shape[1:])

    # Check GPU availability in Colab
    if torch.cuda.is_available():
        device = torch.device("cuda")
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024 * 1024)
        print(f"Available GPU Memory: {free:.2f} MB")

        # Adjust batch size based on available GPU memory
        if free > 8200:
            bs = 4  # Larger batch size for more GPU memory
        else:
            bs = 2  # Smaller batch size for less GPU memory
        print(f"Using batch size: {bs} on GPU")
    else:
        device = torch.device("cpu")
        bs = 1  # Safe default for CPU
        print("GPU not available. Falling back to CPU.")
        print(f"Using batch size: {bs} on CPU")

    # Use the device in your model or training loop
    print(f"Using device: {device}")

    # Define data loaders
    datablock = DataBlock(blocks=(ImageBlock(cls=PILImage), MaskBlock(codes=['wall', 'not-wall'])),
                        get_items=get_image_files,
                        splitter=RandomSplitter(valid_pct=0.1, seed=42),
                        get_y=lambda x: path_lbl/f'{x.stem}.png',
                        batch_tfms=[Normalize.from_stats(*imagenet_stats)],
                        item_tfms=[Resize(1024, 1024)])

    dls = datablock.dataloaders(path_img, bs=bs, num_workers=0)  # Set num_workers=0 for debugging


    # Define metrics and loss function
    tensmask = tensor(mask)
    elems, counts = torch.unique(tensmask, return_counts=True)
    print(f"Mask pixel counts: {counts}")
    sum = torch.sum(counts).item()
    weights = (1.0 - counts/sum)
    print(f"Class weights: {weights}")
    loss_func = CrossEntropyLossFlat(weight=weights, axis=1)
    metrics = [acc_wall, precision, recall]


    datablock = DataBlock(
        blocks=(ImageBlock(cls=PILImage), MaskBlock(codes=codes)),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.1, seed=42),
        get_y=lambda x: path_lbl / f'{x.stem}.png',
        batch_tfms=[Normalize.from_stats(*imagenet_stats)],
        item_tfms=[Resize(1024, 1024)]
    )
    dls = datablock.dataloaders(path_img, bs=1, num_workers=0)

    learn = unet_learner(dls, resnet34, metrics=metrics, loss_func=loss_func)
    learn.load("models-unet-wall-seg-weights/stage-3")  # Adjust path to your model

    # Run the processing
    process_folder(input_folder, output_folder)