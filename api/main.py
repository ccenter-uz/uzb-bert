# api/main.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from spell_checker.core import SpellChecker
from spell_checker.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Uzbek Spell Checker API",
    description="BERT-based spell checking API for Uzbek language",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization of spell checker
checker: Optional[SpellChecker] = None

def get_checker() -> SpellChecker:
    global checker
    if checker is None:
        checker = SpellChecker()
    return checker

class SuggestRequest(BaseModel):
    text: str = Field(..., description="Text to check for spelling")
    max_suggestions: Optional[int] = Field(
        default=settings.DEFAULT_MAX_SUGGESTIONS,
        ge=1,
        le=10,
        description="Maximum number of suggestions per word"
    )

class BatchSuggestRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to check for spelling")
    max_suggestions: Optional[int] = Field(
        default=settings.DEFAULT_MAX_SUGGESTIONS,
        ge=1,
        le=10,
        description="Maximum number of suggestions per word"
    )

@app.get("/health")
async def health_check():
    try:
        # Verify dictionary and model are accessible
        spell_checker = get_checker()
        _ = spell_checker.dictionary  # Test dictionary loading
        _ = spell_checker.model      # Test model loading
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/suggest")
async def suggest_words(req: SuggestRequest):
    try:
        spell_checker = get_checker()
        result = spell_checker.check(req.text, max_suggestions=req.max_suggestions)
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest/batch")
async def suggest_words_batch(req: BatchSuggestRequest):
    try:
        spell_checker = get_checker()
        results = spell_checker.check_batch(req.texts, max_suggestions=req.max_suggestions)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "name": "Uzbek Spell Checker API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/", "methods": ["GET"], "description": "API information"},
            {"path": "/health", "methods": ["GET"], "description": "Health check"},
            {"path": "/suggest", "methods": ["POST"], "description": "Check spelling for a single text"},
            {"path": "/suggest/batch", "methods": ["POST"], "description": "Check spelling for multiple texts"}
        ]
    }
