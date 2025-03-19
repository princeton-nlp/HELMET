# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

export TAG="helmet"
echo "Building the vllm-gaudi docker images"
git clone https://github.com/HabanaAI/vllm-fork.git
cd ./vllm-fork
git checkout v0.6.4.post2+Gaudi-1.19.1 #habana_main

docker build --no-cache -f Dockerfile.hpu -t ${REGISTRY:-opea}/vllm-gaudi:${TAG:-latest} --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
if [ $? -ne 0 ]; then
    echo "vllm-gaudi failed"
    exit 1
else
    echo "vllm-gaudi successful"
fi


