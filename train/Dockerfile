FROM python:3.8-slim-buster

WORKDIR /app

ADD requirements.txt /app

RUN apt-get update -y
RUN apt-get install gcc python3-dev -y
RUN pip install --no-cache-dir -r requirements.txt

ADD . /app

# Run split.py when the container launches
CMD ["python", "main.py"]