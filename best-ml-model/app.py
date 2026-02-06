from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.route("/upload-file", methods=["GET","POST"])
def upload_file(self):

    if request.method == "POST":

        file = request.files["file"]

        if file.filename == "":
            return "No file selected!"

        if file and file.filename.endswith(".csv"):

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Read CSV
            self.df = pd.read_csv(filepath)

            return f"""
            File Uploaded Successfully! 🎉 <br>
            Rows: {self.df.shape[0]} <br>
            Columns: {self.df.shape[1]}
            """

    return render_template("upload.html")



@app.route("/", methods=["GET"])
def welcome():
    return ""

@app.route("/simple-linear",method=["GET","POST"])
def linear_model(self.df):
    if self.df.isnull().sum().sum() == 0:
        