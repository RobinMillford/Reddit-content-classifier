# UPDATED: Changed from python:3.9-slim to python:3.11-slim
# This ensures your deployed application uses the same Python version as your build process.
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# This command tells the container what to run when it starts.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
