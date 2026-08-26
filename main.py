"""
FastAPI service that:
1. Accepts a JSON payload via POST
2. Calls a third-party LLM API (e.g. OpenAI-compatible endpoint)
3. Receives HTML content back
4. Saves it to a local file
5. Returns the saved file path (and optionally the HTML itself) to the caller

Run with:
    uvicorn main:app --reload --port 8000

Env vars required:
    THIRD_PARTY_API_URL   e.g. https://api.openai.com/v1/chat/completions
    THIRD_PARTY_API_KEY   your API key
"""

import os
import uuid
import logging
from pathlib import Path
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

THIRD_PARTY_API_URL = os.getenv("THIRD_PARTY_API_URL", "https://api.openai.com/v1/responses")
THIRD_PARTY_API_KEY = os.getenv("THIRD_PARTY_API_KEY", "")
OUTPUT_DIR = Path(os.getenv("HTML_OUTPUT_DIR", "./generated_html"))
REQUEST_TIMEOUT = 60.0  # seconds

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("html-generator")

app = FastAPI(title="LLM HTML Generator")


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class GenerateHtmlRequest(BaseModel):
    prompt: str = Field(..., description="Instruction describing the HTML to generate")
    model: str = Field(default="gpt-4o-mini", description="Model name to call")
    filename: str = Field(default="component", description="Optional custom filename (without extension)")
    extra_params: dict  = Field(default=dict(), description="Any extra params to pass to the third-party API")


class GenerateHtmlResponse(BaseModel):
    success: bool
    file_path: str
    file_name: str
    size_bytes: int
    created_at: str


# --------------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------------

async def call_third_party_api(payload: GenerateHtmlRequest) -> bytes:
    """
    Calls the third-party LLM API and returns raw HTML content as bytes.
    Adjust the request body / response parsing to match the actual API you're using.
    """
    if not THIRD_PARTY_API_KEY:
        raise HTTPException(status_code=500, detail="THIRD_PARTY_API_KEY is not configured")

    headers = {
        "Authorization": f"Bearer {THIRD_PARTY_API_KEY}",
        "Content-Type": "application/json",
    }

    # Example body for an OpenAI-style chat completion call.
    # Adjust this to match whatever third-party LLM API you're actually calling.
    body = {
        "model": payload.model,
        "input": [
            {
                "role": "system",
                "content": "You generate complete, valid HTML documents only. "
                            "Return raw HTML — no markdown fences, no commentary.",
            },
            {"role": "user", "content": payload.prompt},
        ],
    }
    if payload.extra_params:
        body.update(payload.extra_params)

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            resp = await client.post(THIRD_PARTY_API_URL, headers=headers, json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"Third-party API error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Third-party API returned an error: {e.response.status_code}",
            )
        except httpx.RequestError as e:
            logger.error(f"Third-party API request failed: {e}")
            raise HTTPException(status_code=502, detail="Failed to reach third-party API")

    data = resp.json()

    # --- Extract HTML content from the response ---
    # This depends entirely on the third-party API's response shape.
    # Example for an OpenAI-style chat completion response:
    try:
        # html_content = data["choices"][0]["message"]["content"]
        html_content = data["output"][0]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        # Fallback: if the API returns raw bytes/content directly instead of JSON
        raise HTTPException(status_code=502, detail="Unexpected response format from third-party API")

    # Strip accidental markdown code fences if the model wrapped the HTML
    html_content = html_content.strip()
    if html_content.startswith("```"):
        html_content = html_content.split("\n", 1)[-1]
        if html_content.endswith("```"):
            html_content = html_content.rsplit("```", 1)[0]

    return html_content.encode("utf-8")


def save_html_locally(html_bytes: bytes, filename: str) -> Path:
    """Saves HTML bytes to disk and returns the file path."""
    safe_name = filename or f"page_{uuid.uuid4().hex[:8]}"
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("-", "_")) or "page"
    file_path = OUTPUT_DIR / f"{safe_name}.jsx"

    with open(file_path, "wb") as f:
        f.write(html_bytes)

    return file_path


# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------

@app.post("/generate-html", response_model=GenerateHtmlResponse)
async def generate_html(payload: GenerateHtmlRequest):
    """
    Accepts a JSON payload, calls the third-party LLM API, saves the
    returned HTML to a local file, and returns metadata about the saved file.
    """
    html_bytes = await call_third_party_api(payload)
    file_path = save_html_locally(html_bytes, payload.filename)

    return GenerateHtmlResponse(
        success=True,
        file_path=str(file_path.resolve()),
        file_name=file_path.name,
        size_bytes=len(html_bytes),
        created_at=datetime.utcnow().isoformat(),
    )


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})