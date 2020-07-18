#!/bin/bash

# Assume you execute this script in /path/to/libtorch_samples/datasets
if ! [[ -d "./downloaded" ]]; then
    mkdir downloaded
fi
cd downloaded

if ! [[ -d "cifar-10-batches-bin/" ]]; then
	wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
	tar xvzf cifar-10-binary.tar.gz
	rm -f cifar-10-binary.tar.gz
fi

cd ..