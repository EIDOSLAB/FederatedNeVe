#!/bin/sh

docker build -t eidos-service.di.unito.it/dalmasso/federated-neve:base . -f Dockerfile
docker push eidos-service.di.unito.it/dalmasso/federated-neve:base

docker build -t eidos-service.di.unito.it/dalmasso/federated-neve:python . -f Dockerfile.python
docker push eidos-service.di.unito.it/dalmasso/federated-neve:python

docker build -t eidos-service.di.unito.it/dalmasso/federated-neve:sweep . -f Dockerfile.sweep
docker push eidos-service.di.unito.it/dalmasso/federated-neve:sweep