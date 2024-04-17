FROM python:3.10.13

RUN pip3 install --upgrade pip

RUN pip3 install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Add the eidoslab group to the image
# not sure it is really needed but ok
RUN addgroup --gid 1337 eidoslab

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

EXPOSE 6789