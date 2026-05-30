#base image (Use light weight python image)
FROM python:3.12.3-slim

#Prevent python create cache file 
ENV PYTHONDONTWRITEBYTECODE=1

#Show logs instantly
ENV PYTHONUNBUFFERED=1

#Set working directory inside a container
WORKDIR /app

#Install Linux dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements first
COPY requirement.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirement.txt

# Copy all project files
COPY . .

# Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0"]