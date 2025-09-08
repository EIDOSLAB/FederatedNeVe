FROM python:3.11.11-slim

ENV PIP_DEFAULT_TIMEOUT=100 \
    # Prevents python from writing pyc files to disc
    PYTHONDONTWRITEBYTECODE=1 \
    # Allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # Disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1

RUN pip3 install --upgrade pip

RUN pip3 install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu118

# Add the eidoslab group to the image
# not sure it is really needed but ok
RUN addgroup --gid 1337 eidoslab

RUN mkdir /fl
RUN chmod 775 /fl
RUN chown -R :1337 /fl

RUN mkdir /fl/src
RUN chmod 775 /fl/src
RUN chown -R :1337 /fl/src

RUN mkdir /scratch
RUN chmod 775 /scratch
RUN chown -R :1337 /scratch

RUN mkdir /.cache
RUN chmod 775 /.cache
RUN chown -R :1337 /.cache

RUN mkdir /.config
RUN chmod 775 /.config
RUN chown -R :1337 /.config

COPY src /fl/src

COPY requirements.txt /fl
COPY setup.py /fl

WORKDIR /fl

RUN pip3 install -r requirements.txt

RUN pip3 install -e .

WORKDIR /fl/src