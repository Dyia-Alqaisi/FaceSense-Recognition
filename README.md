# FaceSense: Hybrid Wavelet-SVM Recognition System

This project implements an end-to-end facial recognition and verification workflow using the **PINS Face Recognition dataset**. It combines traditional Computer Vision (Haar Cascades) with advanced Signal Processing (Discrete Wavelet Transforms) and Machine Learning (SVM) to classify individuals as `Approved` or `Denied`.

* **Dataset Source:** [PINS Face Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition/data)
* **Project Documentation:** [📄 **Download Full Project Report (PDF)**](ProjectDocumentation.pdf)
* **Core Technologies:** OpenCV, PyWavelets, Scikit-Learn

---

## 🚀 What This Project Does

The workflow is consolidated into a single, high-performance pipeline:

### 1. Data Cleaning & Facial ROI Extraction
A quality-focused preprocessing stage that filters out low-quality images.
* ✅ **Face & Eye Detection:** Uses Haar Cascades to locate faces. An image is only accepted if it contains a face and **at least two eyes**.
* ✅ **Metadata Alignment:** Uses Regular Expressions to normalize celebrity folder names and match them with labels in an external `dataset.xlsx`.
* ✅ **Automated Directory Management:** Cleans old data and organizes successful crops into a structured folder hierarchy.



### 2. Feature Engineering (The Hybrid Approach)
Instead of relying solely on pixels, this project extracts structural and frequency-domain features to improve model robustness.
* **Spatial Data:** Resizes raw face crops to 32×32 across 3 color channels (RGB).
* **Frequency Data:** Applies a **2D Discrete Wavelet Transform (DWT)** using the `db1` (Daubechies) wavelet at level 5.
* **Detail Extraction:** By zeroing out the low-frequency components, the model focuses on high-frequency "details" like edges and facial contours, reducing sensitivity to lighting variations.
* **Feature Stacking:** Concatenates both raw and wavelet data into a unified, high-dimensional feature vector.



### 3. Machine Learning & Pipeline Optimization
Trains a robust classifier using a modular Scikit-Learn architecture.
* **Pipelines:** Encapsulates `StandardScaler` and `SVC` (Support Vector Classifier) into a single object to prevent data leakage.
* **Hyperparameter Tuning:** Uses `GridSearchCV` with **Stratified 5-Fold Cross-Validation** to find the optimal $C$ and $\gamma$ parameters for the RBF kernel.
* **Imbalance Handling:** Uses `class_weight='balanced'` to ensure accurate classification even if the dataset has uneven sample distributions.
* **Persistence:** Saves the entire pipeline (scaler + model) as `svm_baseline.joblib` for easy deployment.



---

## 📂 Project Files

| File | Description |
| :--- | :--- |
| `FaceRecognition_Wavelet_SVM_Pipeline.ipynb` | The complete end-to-end pipeline from raw data to saved model |
| `ProjectDocumentation.pdf` | Detailed technical report including methodology and results |
| `dataset.xlsx` | Metadata and status labels for the image dataset |

---

## 🛠️ Requirements

* Python 3.8+
* OpenCV (`cv2`)
* PyWavelets (`pywt`)
* Scikit-Learn
* Pandas / NumPy
* Joblib / Openpyxl

Install all dependencies via pip:
```bash
pip install opencv-python pywavelets scikit-learn pandas numpy joblib openpyxl
```

## 📝 License

This project code is open source under the **MIT License**.

However, the **PINS Face Recognition Dataset** used in this project is subject to the **CC BY-NC-SA 4.0** license.
* **Code:** Free to use for personal and commercial projects.
* **Dataset:** Attribution required, Non-commercial use only, Share Alike.

