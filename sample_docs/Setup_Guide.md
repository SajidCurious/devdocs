# TaskFlow Setup Guide

## Prerequisites

Install Python 3.12 and Git. Create and activate a virtual environment before installing dependencies.

## Install dependencies

Run `pip install -r requirements.txt` from the project root.

## Configure the API client

Set `TASKFLOW_TIMEOUT_SECONDS` to change the request timeout. If it is not set, this guide says the default request timeout is 60 seconds.

## Run locally

Run `python -m uvicorn taskflow.main:app --reload` and open `http://localhost:8000/docs` to view the API documentation.
