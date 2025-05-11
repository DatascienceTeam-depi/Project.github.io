import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np

# ========== Define the same model class ==========
class EuroSATModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EuroSATModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# ========== Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EuroSATModel(num_classes=10)
model.load_state_dict(torch.load(r"C:\Users\Qaiaty store\Desktop\Project.github.io-main\Project.github.io-main\eurosat_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# ========== Class names ==========
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# ========== Image Transform ==========
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to [-1, 1]
])

# ========== Streamlit UI ==========
st.set_page_config(page_title="EuroSAT Classifier", layout="centered")
st.title("🌍 EuroSAT Land Cover Classifier")
st.write("Upload a satellite image and get the predicted land cover type.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Apply transforms
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
    st.success(f"✅ **Prediction:** {class_name}")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")
