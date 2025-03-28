DeepGazeIIE Model for Saliency Map Prediction
Overview
This repository contains code for utilizing the DeepGazeIIE model to generate saliency maps for images and evaluate them using multiple saliency metrics. The model is applied to the Saliency4ASD dataset, which consists of ASD (Autism Spectrum Disorder) and TD (Typically Developing) fixation maps. The generated saliency maps are compared to fixation maps using various evaluation metrics.
Requirements
To run this code, ensure you have the following dependencies installed:
* Python 3.6+
* Google Colab (recommended for execution)
* Torch
* Torchvision
* NumPy
* SciPy
* Matplotlib
* H5Py
* OpenCV
* scikit-learn
* Pillow
* Pandas
Setup Instructions
1. Upload Dataset
Upload the Saliency4ASD.zip file to the Colab environment:
from google.colab import files
uploaded = files.upload()
2. Extract Dataset
Unzip the dataset to access images and fixation maps:
import zipfile
with zipfile.ZipFile('Saliency4ASD.zip', 'r') as zip_ref:
    zip_ref.extractall('Saliency4ASD')
3. Clone DeepGaze Repository
Clone the DeepGaze model repository and install dependencies:
git clone https://github.com/matthias-k/DeepGaze.git
cd DeepGaze
pip install -r requirements.txt
4. Download Pretrained Model
wget https://github.com/matthiask/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
Running the Model
1. Load Pretrained Model
import torch
import deepgaze_pytorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
2. Generate Saliency Maps
from deepgaze_pytorch import DeepGazeIIE
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.special import logsumexp

def preprocess_image(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0).to(DEVICE)

def load_centerbias():
    return np.load('centerbias_mit1003.npy')

def generate_saliency_map(model, image):
    image_tensor = preprocess_image(image)
    centerbias = load_centerbias()
    centerbias_rescaled = zoom(centerbias, (image_tensor.shape[2] / centerbias.shape[0],
                                            image_tensor.shape[3] / centerbias.shape[1]),
                               order=0, mode='nearest')
    centerbias_rescaled -= logsumexp(centerbias_rescaled)
    centerbias_tensor = torch.tensor([centerbias_rescaled]).to(DEVICE)
    with torch.no_grad():
        saliency_map = model(image_tensor, centerbias_tensor)
    return saliency_map
3. Run the Model on a Dataset
import os
from PIL import Image

def process_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('RGB')
            saliency_map = generate_saliency_map(model, image)
            save_saliency_map(saliency_map, filename)
image_directory = '/content/Saliency4ASD/Images'
process_images(image_directory)
4. Save and Download Saliency Maps
import os
import zipfile

def save_saliency_map(saliency_map, image_name):
    save_path = f"/content/saliency_maps/{image_name}"
    plt.imshow(saliency_map.squeeze().cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
    return save_path

def download_saliency_maps():
    zip_path = "/content/saliency_maps.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir("/content/saliency_maps"):
            if file.endswith('.png'):
                zipf.write(os.path.join("/content/saliency_maps", file))
    from google.colab import files
    files.download(zip_path)Evaluation Metrics
To evaluate the generated saliency maps, the following metrics are implemented:
* AUC (Area Under Curve) Borji
* AUC Judd
* AUC Shuffled
* CC (Correlation Coefficient)
* EMD (Earth Mover’s Distance)
* KL-Divergence
* NSS (Normalized Scanpath Saliency)
* Information Gain
Example Usage
from sklearn.metrics import roc_auc_score
def calculate_auc(y_true, y_pred):
    return roc_auc_score(y_true.flatten(), y_pred.flatten())Running Full Pipeline
f __name__ == '__main__':
    directory_path = '/content/Saliency4ASD/Images'
    process_images(directory_path)
    download_saliency_maps()
Citation
If you use this code, please cite the original DeepGazeIIE model and Saliency4ASD dataset:
* DeepGazeIIE: https://github.com/matthias-k/DeepGaze
* Saliency4ASD Dataset:  Duan H, Zhai G, Min X, Che Z, Fang Y, Yang X, Gutiérrez J, Callet PL. A dataset of eye movements for the children with autism spectrum disorder. InProceedings of the 10th ACM Multimedia Systems Conference 2019 Jun 18 (pp. 255-260).

Acknowledgments
Special thanks to the developers of DeepGazeIIE and Saliency Benchmarking tools for their contributions to saliency research.

