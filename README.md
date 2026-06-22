🏠 House Price Prediction (Full-Stack)

A production-ready full-stack web application that predicts real-time estate prices using Machine Learning and an automated data pipeline.

## ⚡ Quick Features
* **High Reliability:** Engineered with advanced outlier filtering and robust regression models.
* **Automated ML Pipeline:** End-to-end data processing and model inference using Scikit-learn.
* **Modern Stack:** React.js frontend communicating seamlessly with a Flask (Python) REST API.

---

🛠️ Tech Stack
* **Machine Learning / Data Science:** Python, Pandas, NumPy, Scikit-learn (`RandomForestRegressor`), OneHotEncoder, Joblib
* **Backend:** Flask, RESTful API, OS, Re
* **Frontend:** React.js, Axios, Tailwind CSS (optional)

---

📂 Project Structure
```text
├── backend/
│   ├── app.py               # Flask application server
│   ├── model.pkl            # Trained RandomForest model
│   ├── vectorizer/encoder   # OneHotEncoder / Scaler files
│   └── requirements.txt     # Python dependencies
└── frontend/
    ├── src/                 # React components & styling
    ├── package.json         # Node.js dependencies
    └── README.md
