# Gunakan image resmi Python
FROM python:3.10

# Set direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
