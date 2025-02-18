FROM docker.io/python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN groupadd -r -g 1000 admin && \
    useradd -r -u 1000 -g admin admin

RUN mkdir -p /app && \
    chown -R admin:admin /app

RUN pip install vic2png

USER admin
WORKDIR /app

ENTRYPOINT ["vic2png"]