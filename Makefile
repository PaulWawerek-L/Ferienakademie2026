# Ferienakademie 2026 -- container workflow
#
# Everything is CPU-only and multi-arch: students run this on their own
# laptops. See README.md for what that costs and what it buys.

SHELL := /bin/bash
DOCKER ?= docker
COMPOSE ?= $(DOCKER) compose

.DEFAULT_GOAL := help
.PHONY: help sources base build weights smoke experiments uf-bitstream uf-video-bitstream smoke-% shell-% clean-outputs push

help: ## Show this help
	@grep -hE '^[a-zA-Z_%-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sources: ## Clone the four upstream repos into third_party/ (editable, IDE-navigable)
	bash scripts/fetch_sources.sh

base: ## Build the shared PyTorch base image (do this first)
	$(DOCKER) build -t ferienakademie/base:cpu -f docker/base/Dockerfile docker/base

# image-compression first: 01 and 02 share one image, and building both at once
# races on the tag under Docker Engine's containerd image store ("already exists").
build: sources base ## Build every project image
	$(COMPOSE) build image-compression
	$(COMPOSE) build

weights: ## Download pretrained weights into ./weights
	bash scripts/fetch_weights.sh

smoke: ## Run every project's smoke test and print a pass/fail table
	bash scripts/smoke_all.sh

experiments: ## Run all six experiments and regenerate every figure in results/
	bash scripts/experiments/run_all.sh

uf-bitstream: ## Real DCVC-UF-Intra bitstreams on CPU: bytes vs estimate vs VTM
	$(COMPOSE) run --rm --no-deps image-compression bash -lc \
	  'cd /opt/DCVC && python /work/scripts/bitstream/uf_bitstream_demo.py \
	   && python /work/scripts/experiments/plot_rd.py bitstream'

uf-video-bitstream: ## Real DCVC-UF video bitstreams on CPU (HTS + LD, 64 frames)
	$(COMPOSE) run --rm --no-deps video-compression bash -lc \
	  'cd /opt/DCVC && python /work/scripts/bitstream/uf_video_bitstream_demo.py \
	   && python /work/scripts/experiments/plot_rd.py video-bitstream'

smoke-%: ## Run one project's smoke test (e.g. make smoke-style-transfer)
	bash scripts/smoke_all.sh $*

shell-%: ## Open a shell in one project's container (e.g. make shell-style-transfer)
	$(COMPOSE) run --rm $* bash

clean-outputs: ## Delete generated results (keeps images and weights)
	rm -rf outputs/* && echo "outputs/ cleared"
