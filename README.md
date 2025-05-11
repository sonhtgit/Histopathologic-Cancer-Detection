Histopathologic Cancer Detection

This repository contains a deep learning pipeline for classifying histopathologic scans of lymph node sections as cancerous or non-cancerous, developed for the [Kaggle Histopathologic Cancer Detection competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection).

---

🧪 Objective

The goal is to build a binary classifier that can identify metastatic cancer in histopathologic tissue images from the CAMELYON16 dataset. This is a crucial step in automating cancer diagnostics using computer vision.

---

📁 Dataset

- **Source**: [Kaggle competition dataset](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)
- **Format**: RGB `.tif` images of size `96x96`
- **Target**: `label` = 1 (cancer), 0 (no cancer)
- **Location**: When run on Kaggle, data is automatically mounted at:

```python
DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"

🧰 Project Structure
Histopathologic-Cancer-Detection/
├── Histopathologic-Cancer-Detection.ipynb
├── submission.csv
├── README.md


🔍 Methodology
✅ Pretrained ResNet50 architecture (transfer learning)

✅ Layer freezing + unfreezing (layer4 + head fine-tuned)

✅ Weighted sampling to handle class imbalance

✅ Binary Cross Entropy Loss + AUC monitoring

✅ Early stopping & learning rate scheduling

✅ Weight decay sweep ([1e-5, 1e-4, 1e-3]) for regularization tuning

🏁 Training & Inference Workflow
Load images and labels

Create PyTorch datasets/loaders with augmentations

Load ResNet50 (torchvision.models.resnet50) and customize head

Train and evaluate using BCELoss and AUC

Predict on test set and generate submission.csv

📈 Results

Achieved a private leaderboard AUC of 0.9492

Public leaderboard AUC: 0.9569


📝 Submission Format
csv
Copy
Edit
id,label
f38a6374c348f90b587e046aac6079959adf3835,0
c18f2d887b7ae4f6742ee445113fa1aef383ed77,1
...
🚀 Run on Kaggle
This notebook is designed to run fully within the Kaggle environment.

Dataset path: "/kaggle/input/histopathologic-cancer-detection"

GPU acceleration recommended

Model weights and outputs saved to "/kaggle/working/"

📚 Acknowledgments
Kaggle competition organizers

torchvision

CAMELYON16 dataset

📌 Author
Son Ho
@son.ho5411