# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . /app

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501"]