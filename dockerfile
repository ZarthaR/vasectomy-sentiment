# Gunakan image Python yang sesuai
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Salin requirements.txt ke dalam container
COPY requirements.txt .

# Instal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke dalam container
COPY . .

# Perintah untuk menjalankan aplikasi Anda
CMD ["python", "your_script.py"]  # Ganti dengan nama file Python Anda