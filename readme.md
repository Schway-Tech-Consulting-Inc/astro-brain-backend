# Astro Brain Backend

Backend service for a custom GPT that performs astrological calculations.

## Tech stack

- Python
- FastAPI
- Uvicorn
- Skyfield (for astronomical calculations)

## How this will be used

- Code is stored here in GitHub.
- Railway will deploy this repo as a web service.
- A custom GPT will call the backend's HTTP endpoints (like `/chart`) using an OpenAPI schema.

## Basic endpoints (current)

- `GET /` → simple message
- `GET /health` → health check
