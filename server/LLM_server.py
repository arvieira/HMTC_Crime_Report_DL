import uvicorn
import threading

from fastapi import FastAPI, HTTPException
from .generation_request import GenerationRequest
from .generation_response import GenerationResponse
from .model_manager import ModelManager
from contextlib import asynccontextmanager


# Create ded model manager and load LLM on VRAM
@asynccontextmanager
async def lifespan(application: FastAPI):
    app.state.manager = ModelManager()
    yield


# FastAPI application
app = FastAPI(title="Transformer Server API", lifespan=lifespan)


# Endpoint for LLM generation
@app.post("/v1/completions", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    try:
        app.state.manager.verify_model(request.model)
        
        inputs = app.state.manager.prepare_prompt(request.prompt, request.model)
        generated = app.state.manager.generate_text(inputs, request.model, request.max_tokens, request.temperature)

        return GenerationResponse(
            response_id=request.request_id,
            generated_text=generated['generated_text'],
            token_probabilities=generated['probs'],
            classification=generated['classification'],
            full_prob=generated['full_prob'],
            class_prob=generated['class_prob']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to status verification
@app.get("/ready")
def ready():
    return {"status": "ok"}


# Class to start the server
class LLMServer:
    # Constructor
    def __init__(self, host="127.0.0.1", port=8053):
        self.host = host
        self.port = port
        self.thread = None
        self.server = None
        self.ready = False

    # Start method
    def start(self):
        if self.thread is None:
            config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
            self.server = uvicorn.Server(config)

            self.thread = threading.Thread(target=self.server.run, daemon=True)
            self.thread.start()

    # Stop method
    def stop(self):
        if self.server and self.thread:
            app.state.manager.stop()
            
            self.server.should_exit = True
            self.thread.join()
            self.thread = None
