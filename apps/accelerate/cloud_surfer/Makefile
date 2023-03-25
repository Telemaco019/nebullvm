DOCKER_REGISTRY ?= telemaco019
SURFER_VERSION ?= 0.1.0

DOCKER_IMAGE_PY319_CU11 ?= $(DOCKER_REGISTRY)/cloud-surfer:$(SURFER_VERSION)-py319-cu118
DOCKER_IMAGE_PY319_CU12 ?= $(DOCKER_REGISTRY)/cloud-surfer:$(SURFER_VERSION)-py319-cu12

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Test
.PHONY: test
test: ## Run the tests
	coverage run -m pytest -v .


##@ Build
.PHONY: docker-build-py319-cu118
docker-build-py319-cu118: ## Build the docker image for Python 3.9, NVIDIA CUDA 11.8
	docker build -t $(DOCKER_IMAGE_PY319_CU11) -f deploy/Dockerfile.py319-cu118 .

.PHONY: docker-build-py319-cu12
docker-build-py319-cu12: ## Build the docker image for Python 3.9, NVIDIA CUDA 12
	docker build -t $(DOCKER_IMAGE_PY319_CU12) -f deploy/Dockerfile.py319-cu12 .

.PHONY: docker-push-py319-cu118
docker-push-py319-cu118: ## Push the docker image for Python 3.9, NVIDIA CUDA 11.8

.PHONY: docker-push-py319-cu12
docker-push-py319-cu12: ## Push the docker image for Python 3.9, NVIDIA CUDA 12
