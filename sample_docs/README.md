# TaskFlow Service

TaskFlow is a sample task management API used for the DevDocs Assistant class demo.

## Local development

Use Python 3.12. Create and activate a virtual environment, install the packages in requirements.txt, then start the web application with `streamlit run app.py`.

## Configuration

The service reads `TASKFLOW_ENV` to select the runtime environment. Supported values are `development`, `staging`, and `production`. The default is `development`.

## Running the API

Start the API with `python -m uvicorn taskflow.main:app --reload`. The default local port is 8000.
