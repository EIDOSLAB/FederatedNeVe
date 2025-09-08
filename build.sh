#!/bin/sh

docker build -t my_registry/fedneve:base . -f Dockerfile
docker push my_registry/fedneve:base

docker build -t my_registry/fedneve:python . -f Dockerfile.python
docker push my_registry/fedneve:python

docker build -t my_registry/fedneve:sweep . -f Dockerfile.sweep
docker push my_registry/fedneve:sweep