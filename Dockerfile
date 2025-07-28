# Use an official, lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container to /app
# All subsequent commands will run from this directory
WORKDIR /app

# Copy the requirements file into the container
# This is done first to leverage Docker's layer caching. 
# The dependencies will only be re-installed if requirements.txt changes.
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (the 'src' folder, etc.) into the container
COPY . .

# This command tells the container what to run when it starts.
# It starts the Uvicorn server for your FastAPI application.
# --host 0.0.0.0 is crucial for deployment as it makes the app accessible from outside the container.
# --port 80 is a standard web port often used by hosting services like Render.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]