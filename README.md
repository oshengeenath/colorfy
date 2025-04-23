# 🎨 Colorfy: Deep Learning for Grayscale Image Colorization

Welcome to **Colorfy**, a research project aimed at enhancing the quality of grayscale image colorization using fine-tuned deep learning models. This project focuses on improving realism and consistency across various grayscale inputs by leveraging custom datasets and training strategies.


📂 **GitHub Repository**: [https://github.com/oshengeenath/colorfy](https://github.com/oshengeenath/colorfy)

📦 **Pre-trained Weights**: [Download Here](https://drive.google.com/file/d/1V1rJtuAAh8nxUvE6mN9VaZZ7sagkhYnx/view?usp=sharing)

---

## 🔧 About the Project

This project explores advanced neural network techniques to transform grayscale images into vibrant, colorized versions. We use supervised training with a curated dataset of human-centric images, focusing particularly on tan skin tone representation (TanVis dataset).

The primary goals of this project are:
- Enhance skin tone accuracy in grayscale-to-color transformations  
- Minimize color bleeding and artifact generation  
- Develop a robust and generalizable colorization model

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/oshengeenath/colorfy.git
cd colorfy
```

### 2. Set Up Environment

Ensure Python 3.9+ is installed. Use `conda` or `venv` to create a virtual environment:

```bash
pip install -r requirements.txt
```

## 🧪 Running the Application
To start inference:
```bash
python app.py
```

## 📁 Dataset
We use the TanVis dataset, a curated set of 5,000 tan-skinned human images collected from diverse internet sources. All images are:
- 256×256 resolution 
- Preprocessed and normalized 
- Manually verified for quality and duplicates


## 📊 Evaluation Metrics
We evaluate model performance using:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index) 
- CF (Colorfulness Metric)









