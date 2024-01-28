FROM python:3.10-slim-bullseye

ENV HOST=0.0.0.0

ENV LISTEN_PORT 8080

EXPOSE 8080

RUN apt-get update && apt-get install -y git

RUN apt-get install -y build-essential
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR app/

COPY ./BasicChain.py /app/
CMD ["python", "BasicChain.py"]