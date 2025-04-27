import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

def run_midas(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MiDaS model
    model_type = "DPT_Hybrid"  # DPT_Large, DPT_Hybrid, or MiDaS_small
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    # Correct manual transform (don't use hubconf transforms)
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    # Transform
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        prediction = midas(input_tensor)

    # Resize prediction to match original image size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map
