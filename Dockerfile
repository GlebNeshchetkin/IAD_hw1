FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir numba
CMD ["python3", "hw1.py"]