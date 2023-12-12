FROM python:3.10.13

RUN pip3 install --upgrade pip

RUN pip3 install install torch==2.1.1 torchvision==0.16.1 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install scikit_learn==1.2.0 wandb==0.13.9

# Add the eidoslab group to the image
# not sure it is really needed but ok
RUN addgroup --gid 1337 eidoslab

RUN mkdir /.config
RUN chmod 775 /.config
RUN chown -R :1337 /.config

RUN mkdir /.cache
RUN chmod 775 /.cache
RUN chown -R :1337 /.cache

RUN mkdir /src
RUN chmod 775 /src
RUN chown -R :1337 /src

RUN mkdir /scratch
RUN chmod 775 /scratch
RUN chown -R :1337 /scratch

COPY src /src

COPY requirements.txt /src

WORKDIR /src/

RUN pip3 install -r requirements.txt