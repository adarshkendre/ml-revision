from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
# import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.route("/")
def index():
    return redirect(url_for("upload_file"))

@app.route("/upload-file", methods=["GET","POST"])
def upload_file():
    global df
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return "No file selected!"

        if file and file.filename.endswith(".csv"):

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Read CSV
            df = pd.read_csv(filepath)
            if df.select_dtypes(include='number').shape[1] != df.shape[1]:
                return "All columns must be numeric."

                # Check null
            if df.isnull().sum().sum() > 0:
                return "Dataset contains missing values."

            return redirect(url_for("results"))
    
    return render_template("upload.html")


@app.route("/results",methods=["GET","POST"])
def results():
    if 'df' not in globals() or df is None:
        return redirect(url_for("upload_file"))
    
    X = df.iloc[:, :-1]   # Features
    y = df.iloc[:, -1]    # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)
    elastic = ElasticNet()

    lr.fit(X_train_scaled,y_train)

    ridge.fit(X_train_scaled,y_train)

    lasso.fit(X_train_scaled,y_train)

    elastic.fit(X_train_scaled,y_train)

    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_ridge = ridge.predict(X_test_scaled)
    y_pred_lasso = lasso.predict(X_test_scaled)
    y_pred_elastic = elastic.predict(X_test_scaled)

    r2_scores = {"r2_lr" : r2_score(y_test,y_pred_lr),
    "r2_ridge" : r2_score(y_test,y_pred_ridge),
    "r2_lasso" : r2_score(y_test,y_pred_lasso),
    "r2_elastic" : r2_score(y_test,y_pred_elastic)}

 
    sorted_results = dict(sorted(r2_scores.items(), key=lambda x: x[1], reverse=True))
    html = "<h3>Model Results (Sorted by R² Score)</h3><table border='1'><tr><th>Model</th><th>R² Score</th></tr>"
    for model, score in sorted_results.items():
        html += f"<tr><td>{model}</td><td>{score:.4f}</td></tr>"
    html += "</table>"
    return html





