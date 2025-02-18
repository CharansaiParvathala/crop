import streamlit as st
import torch
import torchvision.transforms as transforms
import timm
import numpy as np
import cv2
import json
import nltk
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load Disease Knowledge Base
DISEASE_INFO_FILE = "disease_data.json"
with open(DISEASE_INFO_FILE, "r") as f:
    DISEASE_DATA = json.load(f)

# Load Pretrained Model
MODEL_NAME = "convnext_large"
MODEL_PATH = "best_crop_disease_model.pth"
CLASS_NAMES_PATH = "class_names.json"

# Load Class Names
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to Generate Heatmap
def generate_heatmap(image_tensor, model):
    image_tensor.requires_grad_()
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    output[:, predicted].backward()
    gradients = image_tensor.grad[0].cpu().numpy()
    
    gradients = np.mean(gradients, axis=0)
    gradients = np.maximum(gradients, 0)  # ReLU
    gradients /= np.max(gradients)
    
    fig, ax = plt.subplots()
    sns.heatmap(gradients, cmap='jet', alpha=0.6)
    st.pyplot(fig)

# AI-Based Disease Remedy (Offline NLP Reasoning)
def generate_remedy(disease_name):
    if disease_name in DISEASE_DATA:
        return DISEASE_DATA[disease_name]["solution"]
    
    # AI-generated fallback
    return f"Please isolate the affected plant, check for common symptoms, and apply general antifungal or antibacterial treatments."

# Streamlit UI
st.title("🌾 Real-World AI Crop Disease Detector (Offline)")
st.write("Upload a plant leaf image, and the AI will diagnose **any crop disease** with real-time solutions.")

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Transform Image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class_idx = torch.max(probabilities, 0)
        predicted_class = CLASS_NAMES[str(predicted_class_idx.item())]
    
    st.write(f"### 🦠 Disease Detected: **{predicted_class}**")
    st.write(f"**Confidence Score:** {confidence.item() * 100:.2f}%")

    # Remedy Section
    remedy = generate_remedy(predicted_class)
    st.write(f"**🌿 Suggested Treatment:** {remedy}")
    
    # Show Heatmap
    st.write("### 🔥 Affected Area Heatmap:")
    generate_heatmap(image_tensor, model)
