# Use an official, lightweight Python image as the base
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- IMPORTANT CHANGE HERE ---
# Instead of copying everything with "COPY . .", we will be specific.
# This ensures we only copy what the final application needs.

# Copy the source code for the API
COPY ./src ./src

# Explicitly copy the 'mlruns' directory which was generated during the GitHub Action run.
# This is the key fix that ensures your trained models are included in the final image.
COPY ./mlruns ./mlruns

# This command tells the container what to run when it starts.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]