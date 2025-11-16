# ML Project - Git, GitHub & GitHub Actions Demo

A comprehensive demo repository showing Git basics, GitHub collaboration, and GitHub Actions CI/CD with a machine learning project.

## Project Overview

This project implements a simple Logistic Regression classifier for the Iris dataset from scikit-learn.

## Features

- Data loading and preprocessing
- Logistic Regression model training
- Model evaluation and persistence
- Automated testing with pytest
- Code quality checks with flake8
- CI/CD pipeline with GitHub Actions
- Docker containerization

## Repository Structure
ml-app/
├── .github/workflows/ # GitHub Actions workflows
├── src/ # Source code
├── tests/ # Test cases
├── models/ # Trained models
├── requirements.txt # Dependencies
└── README.md

## Git Basics

### Common Git Commands

Clone repository
git clone <repository-url>

Check status
git status

Add files to staging
git add .

Commit changes
git commit -m "Descriptive commit message"

Push to remote
git push origin main

Create and switch to new branch
git checkout -b feature/new-feature

Merge branches
git merge feature/new-feature

text

## GitHub Actions
This repository includes a CI/CD workflow that:
- Runs on push/pull requests to main branch
- Installs dependencies
- Runs flake8 linting
- Executes pytest tests
- Builds Docker image
- Uploads test results and Docker image as artifacts

## Getting Started

1. **Create virtual environment and install dependencies:**
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

2. **Train the model:**
python src/train.py


3. **Run tests:**
pytest tests/

4. **Run linter:**
flake8 src/ tests/


## CI/CD Pipeline

The GitHub Actions workflow automatically:
- Checks code quality
- Runs all tests
- Builds Docker container
- Uploads artifacts

See `.github/workflows/ci.yml` for details.