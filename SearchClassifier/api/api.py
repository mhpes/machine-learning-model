from flask import Flask, request, jsonify
from model import predict_product as model_predict_product
from model import predict_csv as model_predict_csv
from flask_csv import send_csv
from flask.helpers import send_file
from io import StringIO
import pandas as pd
import os
import shutil
TMP_FILES = "tmp"



app = Flask(__name__)

@app.route('/predictProduct/<product>', methods=['GET'])
def predict_product(product):
    print(product)
    prediction = model_predict_product(product)
    return jsonify({'prediction': prediction})

@app.route('/predictCsv', methods=['POST'])
def predict_csv():
    filename=request.form['filename']
    print(filename)
    print(request.form)
    print(request.data)
    print(request.files['data'])
    #data=pd.read_csv(StringIO(request.files['data']), sep=';')
    data = request.files['data'].read() 
    try:
        os.makedirs("tmp")
    except FileExistsError:
    # directory already exists
        pass    
    with open("tmp/data.csv", "wb") as tmp_csv:
        tmp_csv.write(data)
    df = pd.read_csv("tmp/data.csv", sep=';', encoding='cp437', error_bad_lines=False)
    df = df.dropna()
    # columns = data.splitlines()[0].split(b';')
    # print(columns)
    # df = pd.DataFrame(columns=columns)
    # # print(data.splitlines()[1])
    # print("Stripping...")
    # for line in data.splitlines()[1:]:
    #     tokens = line.split(b';')
    #     dict_row = {'kn8': tokens[0], 'product':tokens[1]}
    #     df.append(dict_row, ignore_index=True)
    # print("Stripped...")
    # print(df.head())
    output_csv = model_predict_csv(filename, df)
    resp = send_file(filename_or_fp=output_csv, mimetype="text/csv")
    # Delete file here
    shutil.rmtree(TMP_FILES)
    return resp

if __name__ == "__main__":
    app.run(debug=True, port=5000)