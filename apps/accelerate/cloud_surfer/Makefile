DOCKER_REGISTRY ?= ghcr.io/telemaco019
SURFER_VERSION ?= 0.1.0

DOCKER_IMAGE_PY39_CU11 ?= $(DOCKER_REGISTRY)/cloud-surfer:$(SURFER_VERSION)-py39-cu118

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Build
.PHONY: docker-build-py39-cu118
docker-build-py39-cu118: ## Build the docker image for Python 3.9, NVIDIA CUDA 11.8
	docker build -t $(DOCKER_IMAGE_PY39_CU11) -f deploy/Dockerfile.py39-cu118 .

.PHONY: docker-push-py39-cu118
docker-push-py39-cu118: ## Push the docker image for Python 3.9, NVIDIA CUDA 11.8
	docker push $(DOCKER_IMAGE_PY39_CU11)