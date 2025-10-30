import os
import json
import time
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from setup import LLMService
from util import get_model_dir, trace
from typing import List
from pydantic import BaseModel, Field

# ===============================
# Initialize Service and App
# ===============================
llm = LLMService()

app = FastAPI(
    title="RAG LLM Server",
    description="API for PDF-based RAG pipeline and local LLM chat.",
    version="1.0.0",
    contact={
        "name": "RAG Service",
        "email": "admin@localrag.ai",
    },
    license_info={"name": "MIT"},
)

FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Schemas
# ===============================
class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="The role of the message sender (e.g. 'user', 'assistant', 'system').")
    content: str = Field(..., description="The text content of the message.")


class ChatRequest(BaseModel):
    """Incoming user chat request."""
    messages: List[ChatMessage] = Field(
        ...,
        description="List of chat messages including roles and contents. "
                    "Example: [{'role': 'user', 'content': 'Hello!'}]"
    )

class ChatResponse(BaseModel):
    """Response from the local LLM."""
    response: str


class UploadResponse(BaseModel):
    """Response after uploading a PDF file."""
    message: str


class HealthResponse(BaseModel):
    """Health check status."""
    status: str

# ===============================
# Endpoints
# ===============================

@app.get("/metadata", tags=["Utility"])
def get_metadata():
    return {"metadata": "This is a metadata endpoint."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(response: Response):
    """Check system health status."""
    status = "healthy" if llm.model_loaded else "unhealthy"
    if not llm.model_loaded:
        response.status_code = 503
    return {"status": status}


@app.post(
    "/api/upload_pdf",
    response_model=UploadResponse,
    tags=["Data"],
    description="Upload and process a PDF to build a user-specific retriever."
)
async def upload_pdf(user_id: str, file: UploadFile = File(...)):
    with llm.metrics.tracer.start_as_current_span("upload_pdf") as upload_pdf:
        if not llm.model_loaded:
            raise HTTPException(503, "LLM is still loading. Please wait.")
        try:
            llm.logger.info(f"Updating retriever for user {user_id}...")
            with llm.tracer.start_as_current_span("setup_pipeline", links=[trace.Link(upload_pdf.get_span_context())]):
                llm.agents[user_id] = await llm.setup_pipeline_for_user(user_id=user_id, file=file)
            llm.logger.info("Retriever updated successfully.")
            return {"message": "PDF processed and stored successfully."}
        except Exception as e:
            llm.logger.error(f"Error updating retriever: {e}", exc_info=True)
            raise HTTPException(500, str(e))

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["LLM"],
    description="Chat with the LLM using context from the most recent uploaded PDF."
)
def chat_endpoint(user_id: str, request: ChatRequest):
    llm.metrics.REQUEST_COUNT.inc()
    start_time = time.time()

    if not llm.model_loaded:
        raise HTTPException(503, "LLM is still loading. Please wait.")

    qa_pipeline = llm.agents.get(user_id)
    if qa_pipeline is None:
        raise HTTPException(400, "QA pipeline is not ready. Upload PDF first.")

    if llm.model is None:
        llm.load_model(model_name="Qwen/Qwen2.5-0.5B-Instruct", local_dir=get_model_dir())

    try:
        # Get last user message
        user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
        if not user_message:
            raise HTTPException(400, "No user message provided.")
        response = qa_pipeline.invoke(user_message)
        response_text = response["result"].split("Answer:")[-1].strip()
    except Exception as e:
        llm.logger.error(f"Pipeline error: {e}")
        raise HTTPException(500, "Failed to process request")

    llm.LATENCY.observe(time.time() - start_time)
    return {"response": response_text}



@app.get("/api/config", tags=["System"])
def get_config():
    return {
        "backend_name": "rag-pipeline",
        "models": [{"id": "qwen", "name": "Qwen 2.5 Instruct"}],
    }


@app.get("/metrics", description="Prometheus metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ===============================
# Utility
# ===============================
def load_llm(model_name="Qwen/Qwen2.5-0.5B-Instruct", llm: LLMService = None):
    """Load the LLM model at startup."""
    start_time = time.time()
    if llm is None:
        raise Exception("No LLM service provided")

    try:
        llm.load_model(local_dir=get_model_dir(model_name), model_name=model_name)
        llm.metrics.MODEL_LOAD_TIME.observe(time.time() - start_time)
        if not llm.model_loaded:
            raise Exception("Model failed to load.")
        llm.logger.info("LLM Model Loaded Successfully")
    except Exception as e:
        llm.logger.error(f"LLM Model Load Failed: {e}", exc_info=True)

def generate_openapi_schema():
    """Generate and save the OpenAPI schema to a file."""
    openapi_schema = app.openapi()
    with open("openapi_schema.json", "w") as f:
        json.dump(openapi_schema, f, indent=2)
    return {"message": "OpenAPI schema generated and saved to openapi_schema.json"}


# ===============================
# Run
# ===============================
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Run Local RAG LLM FastAPI Server")
    parser.add_argument('--port', type=int, default=8000, help="Server port")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help="Model to load")
    args = parser.parse_args()

    load_llm(llm=llm)
    threading.Thread(target=llm.metrics.monitor_memory_usage, daemon=True).start()
    generate_openapi_schema()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
