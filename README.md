# Uzbek Spell Checker API

BERT-based spell checking API for the Uzbek language, optimized for production deployment.

## Features

- BERT-based contextual spell checking
- Lazy model loading for efficient resource usage
- Batch processing support
- Production-ready with health checks and error handling
- Docker containerization
- Comprehensive logging
- API documentation with OpenAPI/Swagger

## Quick Start

### Local Development

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Run the FastAPI application:
```bash
uvicorn api.main:app --reload
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t uzbek-spell-checker .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 uzbek-spell-checker
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check endpoint
- `POST /suggest` - Check spelling for a single text
- `POST /suggest/batch` - Check spelling for multiple texts in batch

## Configuration

Environment variables (can be set in `.env`):

- `MODEL_NAME` - HuggingFace model name (default: "tahrirchi/tahrirchi-bert-base")
- `MODEL_TOP_K` - Maximum number of predictions (default: 50)
- `MAX_BATCH_SIZE` - Maximum batch size (default: 32)
- `DEFAULT_MAX_SUGGESTIONS` - Default suggestions per word (default: 3)
- `LOG_LEVEL` - Logging level (default: "INFO")

## Project Structure

```
.
├── api/
│   └── main.py           # FastAPI application
├── spell_checker/
│   ├── config.py         # Configuration settings
│   ├── core.py          # Core spell checker implementation
│   └── utils.py         # Utility functions
├── Dockerfile
├── requirements.txt
└── README.md
```

## Production Deployment Notes

1. Model Optimization:
   - Model is downloaded and cached during Docker build
   - Lazy loading prevents unnecessary resource usage
   - Batch processing available for higher throughput

2. Security:
   - Runs as non-root user in Docker
   - CORS middleware configured
   - Input validation with Pydantic

3. Monitoring:
   - Health check endpoint
   - Comprehensive logging
   - Error handling and reporting

4. Scaling:
   - Stateless design
   - Container-ready
   - Environment variable configuration
