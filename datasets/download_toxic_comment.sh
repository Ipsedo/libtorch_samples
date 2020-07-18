#!/usr/bin/env bash

# Assume you execute this script in /path/to/libtorch_samples/datasets
echo "Will create downloaded dir if doen't exist"
if ! [[ -d "./downloaded" ]]; then
    mkdir downloaded
fi
cd downloaded

# Create mnist directory
echo "Will create toxic_comment dir if doen't exist"
if ! [[ -d "./toxic_comment" ]]; then
    mkdir toxic_comment
fi
cd toxic_comment

# Download Toxix comment with Kaggle API
echo "Will download jigsaw-toxic-comment-classification-challenge.zip if doen't exist"
if ! [[ -f "./jigsaw-toxic-comment-classification-challenge.zip" ]]; then
    kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
fi

echo "Unzip all CSVs (overwrite)"
unzip -o jigsaw-toxic-comment-classification-challenge.zip

unzip -o train.csv.zip
unzip -o test.csv.zip
unzip -o test_labels.csv.zip