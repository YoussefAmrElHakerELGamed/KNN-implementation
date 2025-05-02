# **Heart Disease Prediction using K-Nearest Neighbors (KNN)**  
### **Machine Learning Project | Assignment 2**  

---

## **📌 Project Overview**  
This project implements a **K-Nearest Neighbors (KNN)** classifier to predict **heart disease** based on patient health data. The model is trained, validated, and tested using a structured **60-20-20 split**, with hyperparameter tuning and cross-validation to ensure robustness.  

### **📂 Dataset**  
- **File:** `Heart Prediction Quantum Dataset.csv`  
- **Target Variable:** `HeartDisease` (Binary: `0` = No Heart Disease, `1` = Heart Disease)  
- **Features:** Includes attributes like `age`, `Gender`, `Cholesterol`, `BloodPressure` and `HeartRate`.  

---

## **🚀 Key Steps & Results**  

### **1️⃣ Data Preprocessing**  
- **Handled missing values** (if any) using mean imputation.  
- **Standardized features** (critical for KNN distance calculations).  

### **2️⃣ Train-Validation-Test Split**  
| Dataset      | Samples | Percentage |  
|-------------|---------|------------|  
| **Training** | 60%     | Used for model fitting |  
| **Validation** | 20%   | Used for hyperparameter tuning (selecting best K) |  
| **Test** | 20%      | Final evaluation |  

### **3️⃣ KNN Model Training & Tuning**  
- Tested **K values from 1 to 30** and selected the best based on validation accuracy.  
- **Optimal K:** `best_k = 17` (achieved highest validation accuracy).  

### **4️⃣ Cross-Validation (5-Fold)**  
- **Mean CV Accuracy:** `0.8600`  
- **Standard Deviation:** `0.0291`  

### **5️⃣ Performance Metrics (Test Set)**  
| Metric      | Score |  
|------------|-------|  
| **Accuracy** | `0.8600` |  
| **Precision** | `0.8833` |  
| **Recall** | `0.8833` |  
| **F1-Score** | `0.8833` |  

### **6️⃣ Overfitting Analysis**  
- **Training Accuracy:** `0.8967`  
- **Test Accuracy:** `0.8600`  
- **Difference:** `0.0367` → Model generalizes well (no severe overfitting).  

### **7️⃣ Visualizations**  
✅ **Confusion Matrix Heatmap** (Actual vs. Predicted Classes)  
✅ **K vs. Accuracy Plot** (Hyperparameter Tuning)  
✅ **2D PCA Projection** (Optional, for feature analysis)  

---

## **📂 File Structure**  
```
├── Heart_Prediction_KNN.py  # python file with full code  
├── Heart Prediction Quantum Dataset.csv  # Dataset  
└── README.md  # This file  
```

---

## **🛠️ How to Run the Code**  
1. **Install dependencies:**  
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
2. **Run the Jupyter Notebook:**  
   ```bash
   jupyter notebook Heart_Prediction_KNN.ipynb
   ```
3. **Modify dataset path** (if needed) in the notebook.  

---

## **📝 Key Takeaways**  
✔ **KNN performs well** for heart disease prediction (accuracy: `86.0%`).  
✔ **Optimal K = `17`** balances bias and variance.  
✔ **Cross-validation confirms stability** (low std dev in accuracy).  
✔ **No severe overfitting** (train-test accuracy difference < 10%).  

---

## **📜 License**  
This project is open-source under the **MIT License**.  
