# Use a slim Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies (only if needed)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files first to optimize caching
COPY setup.py .
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose application port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]