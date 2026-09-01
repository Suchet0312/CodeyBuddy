"""
python -m codelens

Starts the CodeLens FastAPI server with uvicorn.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "codelens.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
