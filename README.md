# 💸 Loan Prediction Streamlit App

This is a Machine Learning web app built using Streamlit that predicts whether a loan will be approved or not based on applicant details.

## 🧠 How it works
- The model is trained using `train_model.py`
- You can enter applicant details or upload a CSV in the web app
- The app will predict **Approved ✅** or **Rejected ❌**

## 🚀 How to run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
