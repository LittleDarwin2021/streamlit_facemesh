# Dockerfile
FROM python:3.8


COPY packages.txt .

RUN apt update && apt upgrade -y 
RUN apt install -y libgtk2.0-dev
RUN apt install -y libgl1-mesa-glx
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev
RUN apt install -y libtesseract4
RUN apt install -y tesseract-ocr-all


COPY requirements.txt .

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install notebook

WORKDIR /src
COPY /src /src