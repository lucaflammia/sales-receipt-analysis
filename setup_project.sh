#!/bin/bash

echo "Setting up project directory structure..."

# Create core directories
mkdir -p .devcontainer
mkdir -p .github/workflows
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p notebooks
mkdir -p src
mkdir -p tests

echo "Directory structure created."
echo "Please ensure Docker and VS Code are installed to use DevContainers."
echo "Run 'docker build -t sales-receipt-analysis-devcontainer .' in the project root to build the Docker image, or let VS Code do it automatically."
