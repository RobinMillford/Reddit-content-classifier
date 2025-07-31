# Use an official, lightweight Python image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the source code for the API
COPY ./src ./src

# Command to run the app on port 80, which Render expects
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]