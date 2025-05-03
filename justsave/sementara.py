import pandas as pd

# Baca file utama & file tambahan
df_main = pd.read_csv("komentar_preprocessed.csv")
df_new = pd.read_csv("komentar_negatif_manual.csv")

# Gabungkan
df_combined = pd.concat([df_main, df_new], ignore_index=True)

# Simpan
df_combined.to_csv("komentar_preprocessed_updated.csv", index=False)
print("✅ Data berhasil digabung dan disimpan ke komentar_preprocessed_updated.csv")
