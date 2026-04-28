# Base image
FROM python:3.10-slim

# Prevent Python fromwriting .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#set working directory
WORDIR/app

#Install dependencies  
RUN apt-get update && apt-get install -y \                                                                                                    
    gcc \
    && rm -rf /var/lib/apt/lists/*

#Copy requirements first
COPY requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
    
#Copy project files
COPY . .

#Expose port
EXPOSE 8000
     
# Run using Gunicorn 
CMD ["gunicorn","-w","3","-k","uvicorn.workers.UvicornWorker","app:app","--bind","0.0.0.0:8000"]