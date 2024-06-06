# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.8.12
FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y
ENV LANG en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y git
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt-get install -y python3.8 pip
RUN apt-get install -y python3-sphinx

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /Chess

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python3 -m pip install -r requirements.txt --no-cache-dir

ENV PATH /appuser/.local/bin:$PATH

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
# VOLUME [".:/Chess"]

# Expose the port that the application listens on.
EXPOSE 8000

ENV DEBIAN_FRONTEND=dialog
