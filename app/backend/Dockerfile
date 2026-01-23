FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Copy requirements file from backend folder
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application using Flask's built-in server
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "5001"]
