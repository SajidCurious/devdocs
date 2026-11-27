# TaskFlow API Reference

## Authentication

Send a bearer token in the `Authorization` header for protected endpoints. The public health endpoint does not require authentication.

## Create a task

`POST /api/tasks` accepts a JSON body with a required `title` and an optional `description`. A successful response uses HTTP status 201 and includes the new task ID.

## Request timeout

The API client uses a default request timeout of 30 seconds. A caller can override the timeout with the `TASKFLOW_TIMEOUT_SECONDS` environment variable.

## Rate limits

The API allows 100 requests per minute for each authenticated account. Exceeding the limit returns HTTP status 429.
