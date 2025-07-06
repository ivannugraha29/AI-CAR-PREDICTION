import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("used_car_data_new.csv")
df['usia'] = 2025 - df['year']
df = df[df['usia'] <= 20]

le_brand = LabelEncoder()
df['brand_encoded'] = le_brand.fit_transform(df['id_merk'].astype(str))

# Ubah angka menjadi label teks
transmission_map = {
    1: 'Manual',
    2: 'Automatic'
}
df['transmisi'] = df['id_transmission'].map(transmission_map)

# Encode label teks jadi angka
le_trans = LabelEncoder()
df['transmission_encoded'] = le_trans.fit_transform(df['transmisi'])



X = df[['usia', 'brand_encoded', 'transmission_encoded']]
y = df['price_cash']

# Buat data depresiasi harga rata-rata per usia
avg_price_by_age = df.groupby('usia')['price_cash'].mean().reset_index()
avg_price_by_age.to_csv("static/depresiasi.csv", index=False)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

pickle.dump(model, open("model/model_rf.pkl", "wb"))
pickle.dump(le_brand, open("model/le_brand.pkl", "wb"))
pickle.dump(le_trans, open("model/le_trans.pkl", "wb"))
