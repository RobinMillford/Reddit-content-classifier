# Use an official, lightweight Python image as the base
FROM python:3.11-slim

# Create a non-root user for security, as recommended by Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME="/home/user"

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies as the non-root user
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of your application code
COPY --chown=user . .

# Expose the port the app runs on
EXPOSE 7860

# This command tells the container what to run when it starts.
# It uses the correct port 7860 as required by Hugging Face Spaces.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]