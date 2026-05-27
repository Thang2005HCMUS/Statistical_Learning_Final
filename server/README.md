```
uvicorn server:app --reload --port 8000
```

This command starts the FastAPI server defined in the `server.py` file. The `--reload` flag allows the server to automatically reload when code changes are detected, which is useful during development. The `--port 8000` option specifies that the server will listen on port 8000. You can access the API endpoints at `http://localhost:8000`.