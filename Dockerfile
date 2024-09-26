FROM python:3.8.12
ARG testing=1
ARG documentation=1
ARG browser=1

RUN useradd -m -u 1000 user

RUN apt-get update && apt-get install -y python3-pygame

RUN pip install -U pip setuptools poetry
COPY pyproject.toml .
RUN poetry config virtualenvs.create false && \
poetry install --no-root --no-interaction --no-ansi \
$( if [ "$testing" = "1" ]; then echo "-E testing"; fi) \
$( if [ "$documentation" = "1" ]; then echo "-E documentation"; fi) \
$( if [ "$browser" = "1" ]; then echo "-E browser"; fi)

USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user