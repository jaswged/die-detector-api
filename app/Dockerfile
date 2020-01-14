# Start with a base image
FROM python:3-onbuild

# Copy our application code
WORKDIR /var/app
COPY . .
COPY requirements.txt .

# Fetch app specific dependencies
RUN pip install --upgrade pip
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN ls -l
RUN pip install --no-cache-dir fastai==1.0.60

# Expose port
EXPOSE 5000

# Start the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
