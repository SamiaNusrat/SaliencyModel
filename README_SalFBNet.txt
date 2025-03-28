SalFBNet - Saliency Map Generation and Evaluation

This script generates saliency maps for images using the SalFBNet model and evaluates them against fixation maps from TD (Typically Developing) and ASD (Autism Spectrum Disorder) groups using various metrics.

---

### Code Running Instructions

1. **Prerequisites**:
   - Python 3.6 or later.
   - Required libraries: `torch`, `torchvision`, `numpy`, `scikit-learn`, `scipy`, `opencv-python`, `matplotlib`, `Pillow`, `pandas`, `tqdm`.
   - Ensure CUDA is available if using GPU (`cuda`). The script will default to CPU if CUDA is not available.

2. **Setup**:
   - Clone the repository:
     ```bash
     git clone https://github.com/gqding/SalFBNet.git
     ```
   - Download the dataset and pre-trained model:
     - Place the `Saliency4asd.zip` file in the root directory and extract it.
     - Download the pre-trained model (`FBNet_Res18Fixed_best_model.pth`) and place it in `/SalFBNet/pretrained_models/`.

3. **Install Dependencies**:
   Run the following command to install the required libraries:
   ```bash
   pip install scikit-learn scipy tensorboard tqdm torchSummaryX opencv-python matplotlib Pillow pandas
   ```

4. **Run the Script**:
   Execute the script using Python:
   ```bash
   python salfbnet.py
   ```

5. **Output**:
   - Saliency maps will be saved in `/content/saliency_maps/`.
   - A ZIP file (`saliency_maps.zip`) containing all saliency maps will be generated.
   - Metrics comparing the saliency maps with TD and ASD fixation maps will be saved as `saliency_metrics.csv`.

6. **Notes**:
   - Ensure the input images are placed in `/content/Saliency4asd/Saliency4asd/Images/`.
   - The script will automatically handle resizing and normalization of images.
   - For Google Colab users, the script includes commands to download the results directly. For local execution, manually retrieve the output files from the specified directories.

---

