# Metrik Evaluasi Regresi, Machine Learning, Deep Learning
## 📌 Tabel Metrik Evaluasi 

| **Metrik Evaluasi** | **Regresi** ✅ | **Klasifikasi** ✅ | **Deep Learning** ✅ | **Contoh Algoritma yang Cocok** |
|--------------------|--------------|-------------------|---------------------|---------------------------------|
| **MAE (Mean Absolute Error)** | ✅ | ❌ | ✅ | Linear Regression, Lasso Regression, Neural Networks (Regression) |
| **MSE (Mean Squared Error)** | ✅ | 🔹 | ✅ | Random Forest Regressor, XGBoost Regressor, Deep Learning Regression |
| **RMSE (Root Mean Squared Error)** | ✅ | ❌ | ✅ | Support Vector Regression (SVR), Bayesian Regression, Deep Learning (ANN) |
| **R² Score (Koefisien Determinasi)** | ✅ | 🔹 | ✅ | Ridge Regression, Decision Tree Regressor, MLPRegressor |
| **MAPE (Mean Absolute Percentage Error)** | ✅ | ❌ | ✅ | Time Series Models (ARIMA, LSTM for forecasting), ElasticNet Regression |
| **Accuracy (Akurasi)** | ❌ | ✅ | ✅ | Logistic Regression, Decision Tree, Random Forest, CNN |
| **Precision (Presisi)** | ❌ | ✅ | ✅ | SVM, Naive Bayes, XGBoost (untuk ketidakseimbangan kelas) |
| **Recall (Sensitivity/TPR)** | ❌ | ✅ | ✅ | Random Forest, Gradient Boosting, RNN untuk klasifikasi sekuens |
| **F1 Score** | ❌ | ✅ | ✅ | AdaBoost, CatBoost, BERT (NLP classification) |
| **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** | 🔹 | ✅ | ✅ | Neural Networks (Binary Classification), XGBoost, LightGBM |
| **Log Loss (Logarithmic Loss)** | 🔹 | ✅ | ✅ | Softmax Regression, ANN (Multi-Class Classification) |
| **Cross-Entropy Loss** | ❌ | ✅ | ✅ | CNN, RNN, Transformer untuk klasifikasi gambar dan teks |
| **Dice Coefficient / IoU (Intersection over Union)** | ❌ | ✅ (segmentation) | ✅ | U-Net, Mask R-CNN, YOLO (image segmentation & object detection) |
| **BLEU Score (Bilingual Evaluation Understudy)** | ❌ | ❌ | ✅ (NLP) | Transformer (BERT, GPT, T5), seq2seq LSTM (Machine Translation) |
| **Perplexity** | ❌ | ❌ | ✅ (NLP) | GPT, LSTM, Transformer untuk prediksi teks |
| **Mean IoU (Mean Intersection over Union)** | ❌ | ✅ (segmentation) | ✅ | DeepLabV3, SegNet, PSPNet (Semantic Segmentation) |
| **Top-k Accuracy** | ❌ | ✅ | ✅ | EfficientNet, ResNet, Vision Transformers (ViT) untuk Image Classification |

---

### 📌 **Kesimpulan:**
- **Regresi** 🟢 → Linear Regression, XGBoost Regressor, Neural Networks  
- **Klasifikasi** 🔵 → SVM, Random Forest, CNN, XGBoost  
- **Deep Learning** 🔴 → CNN (Vision), RNN (Time Series & NLP), Transformer (BERT, GPT)  

🚀 **Gunakan metrik yang sesuai dengan tipe model yang digunakan!**
