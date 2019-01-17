.PHONY: all
all: run build
NAME=mytensor

CONTAINER_NAME=mytensor

.PHONY: run-container
run:
	docker run -it $(CONTAINER_NAME) /bin/bash

build:
	docker build --no-cache -f Dockerfile -t $(CONTAINER_NAME) .
