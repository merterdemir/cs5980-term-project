#!/bin/bash

# Cleans the created Bazel Build
cd im2txt
echo "Cleaning the build..."
bazel clean
echo "Cleaned!"
cd -