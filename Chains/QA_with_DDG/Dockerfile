FROM python:3.10-slim-bookworm

ENV HOST=0.0.0.0

ENV LISTEN_PORT 8080

EXPOSE 8080

RUN apt-get update && apt-get install -y git

RUN apt-get install -y build-essential
RUN apt-get install -y sqlite3
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY DDGQAChain.py /app/
COPY duckduckgo_search_wrapper.py /app/
COPY ddg_web_research.py /app/

WORKDIR app/

CMD ["python", "DDGQAChain.py"]