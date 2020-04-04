#!/bin/bash

# Cleans the created Bazel Build
cd im2txt
echo "Cleaning the Bazel build..."
bazel clean
rm -rf /private/var/tmp/_bazel_*
cd ~-

echo "Cleaning the parser build..."
rm -rf parser/parser
echo "Cleaned!"

