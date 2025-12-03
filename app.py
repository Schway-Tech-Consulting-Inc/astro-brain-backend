from fastapi import FastAPI

app = FastAPI(
    title="Astro Brain Backend",
    version="0.1.0",
    description="Backend service that will power the custom GPT for astrology."
)


@app.get("/")
def root():
    return {"message": "Astro Brain Backend is running."}


@app.get("/health")
def health():
    return {"status": "ok"}
