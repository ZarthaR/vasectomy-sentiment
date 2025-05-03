import pandas as pd
import glob

# Ambil semua file CSV yang namanya cocok
csv_files = glob.glob("komentar_twitter_selenium*.csv")

# Gabungkan semua file ke satu DataFrame
all_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Hapus duplikat berdasarkan isi komentar saja (bukan timestamp, dll)
all_data = all_data.drop_duplicates(subset=["komentar"])

# Simpan ke file final
all_data.to_csv("komentar_FINAL.csv", index=False)
print(f"✅ Total komentar unik: {len(all_data)}")
