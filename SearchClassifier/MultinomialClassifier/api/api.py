from flask import Flask, request, jsonify
from model import predict_product as model_predict_product
from model import predict_csv as model_predict_csv
from flask.helpers import send_file
from io import StringIO
import pandas as pd
import os
import shutil
from flask_cors import CORS
HEROKU_PATH = "SearchClassifier/MultinomialClassifier/api/"
TMP_FILES = "tmp/"
TMP_RELATIVE = "tmp/"



app = Flask(__name__)
CORS(app)


@app.route('/predictProduct/<product>', methods=['GET'])
def predict_product(product):
    print(product)
    prediction = model_predict_product(product)
    print(prediction)
    return jsonify({'prediction': prediction})

@app.route('/predictCsv', methods=['POST'])
def predict_csv():
    filename=request.form['filename']
    print(filename) 
    print(request.files['data'])
    data = request.files['data'].read() 
    try:
        os.makedirs(TMP_FILES)
    except FileExistsError:
    # directory already exists
        pass    
    with open(TMP_FILES + "query.csv", "wb") as tmp_csv:
        tmp_csv.write(data)
    df = pd.read_csv(TMP_FILES + "query.csv", sep=';', encoding='cp437', error_bad_lines=False)
    df = df.dropna()
    print("predict")
    output_csv = model_predict_csv(filename, df)
    resp = send_file(filename_or_fp=TMP_RELATIVE + output_csv, mimetype="text/csv")
    # Delete file here
    shutil.rmtree(TMP_FILES)
    return resp

if __name__ == "__main__":
    app.run(debug=True, port=5000)