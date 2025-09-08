#!/bin/sh

docker build -t eidos-service.di.unito.it/dalmasso/federated-neve-selection:base . -f Dockerfile
docker push eidos-service.di.unito.it/dalmasso/federated-neve-selection:base

docker build -t eidos-service.di.unito.it/dalmasso/federated-neve-selection:python . -f Dockerfile.python
docker push eidos-service.di.unito.it/dalmasso/federated-neve-selection:python

docker build -t eidos-service.di.unito.it/dalmasso/federated-neve-selection:sweep . -f Dockerfile.sweep
docker push eidos-service.di.unito.it/dalmasso/federated-neve-selection:sweep