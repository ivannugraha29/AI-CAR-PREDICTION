from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import csv
import os

app = Flask(__name__)

model = pickle.load(open("model/model_rf.pkl", "rb"))
le_brand = pickle.load(open("model/le_brand.pkl", "rb"))
le_trans = pickle.load(open("model/le_trans.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    usia = int(request.form['usia'])
    brand = request.form['brand']
    transmisi = request.form['transmisi']

    try:
        brand_encoded = le_brand.transform([brand])[0]
        trans_encoded = le_trans.transform([transmisi])[0]
    except:
        return render_template("index.html", pred="Brand atau Transmisi tidak dikenal.")

    input_features = np.array([[usia, brand_encoded, trans_encoded]])
    predicted_price = model.predict(input_features)[0]
    pred_rupiah = int(predicted_price)

    # simpan ke riwayat_prediksi.csv
    csv_path = "riwayat_prediksi.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["usia", "brand", "transmisi", "harga_prediksi"])
        writer.writerow([usia, brand, transmisi, pred_rupiah])

    return render_template("index.html", pred=f'Harga Prediksi: Rp {pred_rupiah:,}')

@app.route('/grafik')
def grafik():
    df = pd.read_csv("static/depresiasi.csv")
    
    plt.figure(figsize=(8,5))
    plt.plot(df['usia'], df['price_cash'], marker='o', color='#00b14f')
    plt.title("Grafik Depresiasi Harga Mobil")
    plt.xlabel("Usia Mobil (tahun)")
    plt.ylabel("Harga Rata-Rata (Rp)")
    plt.grid(True)
    plt.tight_layout()

    grafik_path = "static/grafik.png"
    plt.savefig(grafik_path)
    plt.close()

    return render_template("grafik.html", grafik_url=grafik_path)

if __name__ == '__main__':
    app.run(debug=True)
