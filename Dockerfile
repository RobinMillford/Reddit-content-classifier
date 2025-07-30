# Use an official, lightweight Python image as the base
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code AND the mlruns folder
COPY . .

# This command tells the container what to run when it starts.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]