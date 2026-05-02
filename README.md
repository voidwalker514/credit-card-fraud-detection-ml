# credit-card-fraud-detection-ml
End-to-end Credit Card Fraud Detection system using Random Forest, XGBoost &amp; Logistic Regression with SMOTE balancing, real-time scoring API, and an interactive Streamlit dashboard.

# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?style=flat-square)
![RandomForest](https://img.shields.io/badge/Model-RandomForest-orange?style=flat-square)
![SMOTE](https://img.shields.io/badge/Imbalance-SMOTE-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> End-to-end Machine Learning system that detects fraudulent credit card transactions using ensemble models, handles severe class imbalance via SMOTE, and serves predictions through an interactive Streamlit dashboard — built for Data Science, Machine Learning, and Banking Analytics roles.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Live Demo](#-live-demo)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Features](#-features)
- [Folder Structure](#-folder-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Model Results](#-model-results)
- [Dashboard Pages](#-dashboard-pages)
- [Dataset](#-dataset)
- [Interview Prep](#-interview-prep)
- [Author](#-author)

---

## 📖 About the Project

Credit card fraud costs the global economy **billions of dollars every year**. Banks and fintech companies use Machine Learning models to detect suspicious transactions in real time — blocking fraud before it causes financial damage.

This project replicates that real-world system:

- Trains multiple ML classifiers on highly imbalanced transaction data
- Uses **SMOTE** (Synthetic Minority Oversampling Technique) to handle the ~1% fraud rate
- Evaluates models using **Precision, Recall, AUC-ROC, and Average Precision** — the correct metrics for imbalanced datasets
- Exposes a fully interactive **Streamlit dashboard** with live transaction scoring, fraud simulation, and model evaluation visualizations

---

## 🚀 Live Demo

> Run the project yourself in Google Colab — no local setup needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit 1.28 |
| Tunnel (Colab) | Cloudflare Tunnel |
| Environment | Google Colab / Jupyter |

---

## 🏗 Project Architecture

```
Transaction Data
       │
       ▼
Preprocessing (Scale, Encode, Engineer Features)
       │
       ├──────────────────────┐
       ▼                      ▼
SMOTE Balancing       Feature Engineering
(oversample fraud)    (Hour, IsNight, HighAmount)
       │                      │
       └──────────┬───────────┘
                  ▼
        Classification Model
     (RF / XGBoost / Logistic Reg)
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
  LEGITIMATE            FRAUD ALERT
  ✅ Approved           🚨 Blocked
```

**Data Flow:**
`Raw CSV → Cleaning → Scaling → SMOTE → Train/Test Split → Model → Predict → Score → Alert`

---

## ✨ Features

- ✅ Synthetic + real dataset support (works fully offline)
- ✅ Three models trained and compared: Logistic Regression, Random Forest, XGBoost
- ✅ SMOTE oversampling to fix severe class imbalance (99:1 ratio)
- ✅ Full evaluation suite: Confusion Matrix, ROC Curve, Precision-Recall Curve
- ✅ Interactive Streamlit dashboard with 6 pages
- ✅ Live transaction scorer with adjustable threshold
- ✅ Real-time fraud simulation with TP/TN/FP/FN breakdown
- ✅ Feature importance visualization
- ✅ Runs entirely in Google Colab — no local setup needed

---

## 📁 Folder Structure

```
credit-card-fraud-detection-ml/
│
├── app.py                        ← Streamlit dashboard (all 6 pages)
├── main.py                       ← Standalone ML pipeline script
│
├── notebooks/
│   └── fraud_detection.ipynb     ← Full Colab notebook
│
├── outputs/
│   ├── dashboard_final.png       ← Summary dashboard screenshot
│   ├── model_evaluation.png      ← ROC + confusion matrix
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   └── eda_overview.png
│
├── models/
│   ├── fraud_detection_model.pkl ← Saved best model
│   └── scaler.pkl                ← Saved StandardScaler
│
├── data/
│   └── README.md                 ← Dataset source instructions
│
├── requirements.txt              ← All dependencies
└── README.md                     ← This file
```

---

## ⚙️ Installation & Setup

### Option 1 — Google Colab (Recommended)

No local setup needed. Open the notebook in Colab and run all cells.

```python
# Install all dependencies in Colab
!pip install streamlit==1.28.0 imbalanced-learn xgboost scikit-learn \
             pandas numpy matplotlib seaborn --quiet
```

### Option 2 — Local Setup

**Requirements:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection-ml.git
cd credit-card-fraud-detection-ml

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit dashboard
streamlit run app.py

# 5. Or run the standalone ML pipeline
python main.py
```

---

## ▶️ How to Run

### Run ML Pipeline (terminal / Colab)

```bash
python main.py
```

Expected output:
```
✅ Dataset loaded: 10,000 rows
✅ SMOTE applied: balanced classes
Logistic Regression  | AUC: 0.9421 | AP: 0.7832
Random Forest        | AUC: 0.9876 | AP: 0.9201
XGBoost              | AUC: 0.9891 | AP: 0.9344
🏆 Best model: XGBoost
✅ Model saved: fraud_detection_model.pkl
```

### Run Streamlit Dashboard (Colab)

```python
# In Colab — starts dashboard and opens tunnel
import subprocess, threading, time

threading.Thread(
    target=lambda: subprocess.run(["streamlit", "run", "app.py",
        "--server.port", "8501", "--server.headless", "true"]),
    daemon=True
).start()
time.sleep(6)

# Then run cloudflared to get public URL
```

---

## 📊 Model Results

| Model | AUC-ROC | Avg Precision | Fraud Recall | Fraud Precision |
|---|---|---|---|---|
| Logistic Regression | ~0.94 | ~0.78 | ~0.82 | ~0.74 |
| Random Forest | ~0.98 | ~0.92 | ~0.91 | ~0.89 |
| **XGBoost** ⭐ | **~0.99** | **~0.93** | **~0.92** | **~0.91** |

> **Why not just use Accuracy?** With 99% legitimate transactions, a model that predicts "always legitimate" gets 99% accuracy but catches 0 fraud. That's why we use **Recall** (did we catch all fraud?) and **Precision** (are our alerts accurate?).

---

## 🖥 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Overview | Key metrics, class balance, amount distribution |
| 📊 EDA | Fraud by hour, correlation heatmap, boxplots |
| 🤖 Model Training | Train 3 models, compare results, feature importance |
| 📈 Evaluation | Confusion matrix, ROC curve, Precision-Recall curve |
| 🔍 Live Prediction | Enter any transaction → instant fraud score + gauge |
| 🚨 Simulation | Simulate 10–100 transactions, view TP/FP/FN feed |

---

## 📦 Dataset

This project uses **synthetic data** generated to simulate real credit card transactions — no real customer data is used.

The synthetic dataset mirrors the structure of the popular [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):

- **V1–V28**: Anonymised PCA features
- **Amount**: Transaction amount in dollars
- **Time**: Seconds elapsed since first transaction
- **Class**: 0 = Legitimate, 1 = Fraud

To use the real Kaggle dataset:
1. Download `creditcard.csv` from Kaggle
2. Place it in the `data/` folder
3. Replace the `make_data()` call with `pd.read_csv("data/creditcard.csv")`

---

## 🎯 Interview Prep

**Q: Why use SMOTE instead of just oversampling?**
SMOTE creates synthetic samples by interpolating between existing minority class points rather than duplicating them — this prevents overfitting to repeated examples.

**Q: Why is Recall more important than Precision for fraud detection?**
A missed fraud (False Negative) causes real financial loss to customers. A false alarm (False Positive) is inconvenient but recoverable. So catching all fraud (high Recall) is prioritised over avoiding false alarms.

**Q: Why not use Accuracy as the metric?**
With 99% legitimate transactions, a dummy model that always predicts "not fraud" achieves 99% accuracy while catching zero fraud cases. Accuracy is meaningless for imbalanced datasets.

**Q: What is AUC-ROC?**
Area Under the ROC Curve measures the model's ability to distinguish between classes across all thresholds. A score of 1.0 is perfect, 0.5 is random guessing.

**Q: How would you deploy this in production?**
Wrap the model in a FastAPI endpoint, containerise with Docker, deploy on AWS/GCP, connect to a Kafka stream for real-time transaction scoring, and set up monitoring for model drift.

---

## 👤 Author

**Ishwari Belhekar**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [your_linkedin](https://linkedin.com/in/your_linkedin)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

⭐ **If this project helped you, please give it a star!** It helps others find it.
