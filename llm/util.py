import os
import re
import torch
import psutil
import time
import logging
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider

class Logger:
    def __init__(self, name=None, level=logging.INFO, id : int = None):
        """
        Custom logging wrapper around Python's standard logging.Logger.

        Attributes:
            name (str): Unique name for this logger instance.
            logger (logging.Logger): The actual logger object.
        
        Args:
            name (str, optional): Base name of the logger. Defaults to the class name.
            level (int, optional): Logging level (DEBUG, INFO, etc.). Defaults to logging.INFO.
            id (int, optional): Optional unique identifier appended to the logger name.
        """
        # Use a stable unique logger name — avoids collisions
        base = name or self.__class__.__name__
        self.name = f"{base}{f'.{id}' if id is not None else ''}"        
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(level)

        # Optional: prevent double propagation to root logger
        self.logger.propagate = False  

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, msg, *args, **kwargs): self.logger.debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs):  self.logger.info(msg, *args, **kwargs)
    def warn(self, msg, *args, **kwargs):  self.logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs): self.logger.error(msg, *args, **kwargs)

    def __str__(self):
        return f"<Logger {self.name} level={logging.getLevelName(self.logger.level)}>"

class RAGMetrics :
    def __init__(self, log_level=logging.INFO):
        """
        Metrics and tracing manager for a RAG (Retrieval-Augmented Generation) pipeline.

        Attributes:
            logger (Logger): Instance of the custom Logger class for internal logging.
            tracer (opentelemetry.trace.Tracer): OpenTelemetry tracer for distributed tracing.
            REQUEST_COUNT (prometheus_client.Counter): Tracks total number of requests.
            LATENCY (prometheus_client.Histogram): Measures request latency in seconds.
            MODEL_LOAD_TIME (prometheus_client.Histogram): Measures LLM model loading time.
            MEMORY_USAGE (prometheus_client.Gauge): Monitors memory usage of the process in bytes.

        Args:
            log_level (int, optional): Logging level for the internal logger. Defaults to logging.INFO.
        """
        self.logger = Logger(name="metrics", level=log_level, id=id(self))
        
        # Opentelemetry tracing
        self.tracer = self.init_tracing()
        
        # Prometheus metrics
        self.REQUEST_COUNT = Counter("chatbot_requests_total", "Total requests to chatbot")
        self.LATENCY = Histogram("chatbot_request_latency_seconds", "Chatbot request latency")
        self.MODEL_LOAD_TIME = Histogram("chatbot_model_load_time_seconds", "Time to load the local LLM in secods")
        self.MEMORY_USAGE = Gauge("chatbot_memory_usage_bytes", "Memory usage in bytes for chatbot process")
        
   
    def init_tracing(self, service_name: str = "rag-pipeline-service", jaeger_url: str = "http://jaeger:14268/api/traces") -> any:
        """Initialize OpenTelemetry tracing with Jaeger."""
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        trace.set_tracer_provider(provider)

        tracer = trace.get_tracer_provider().get_tracer(service_name, "0.1.0")
        
        jaeger_exporter = JaegerExporter(collector_endpoint=jaeger_url)
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        return tracer 

    def monitor_memory_usage(self, interval: int=5) -> None:
        """Get memory usage of the LLM.

        Args:
            interval (int, optional): Interval between memory usage report. Defaults to 5.
        """
        while True:
            try:
                process = psutil.Process()
                self.MEMORY_USAGE.set(process.memory_info().rss)  # Update Prometheus gauge
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}", exc_info=True)
            time.sleep(interval)
        
    def track_request(self, func) -> None:
        """Decorator to track request count and latency, ignoring metric/tracing errors."""
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # increment safely
            try:
                if hasattr(self, "REQUEST_COUNT"):
                    self.REQUEST_COUNT.inc()
            except Exception:
                pass  # ignore metric errors
                
            # start tracing safely
            try:
                tracer = getattr(self, "tracer", None)
                if tracer:
                    with tracer.start_as_current_span(func.__name__):
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            except Exception:
                self.logger.warn("issue pushing trace to jaeger")
                result = func(*args, **kwargs)
            try:
                if hasattr(self, "LATENCY"):
                    elapsed = time.time() - start_time
                    self.LATENCY.observe(elapsed)
            except Exception:
                pass

            return result
        return wrapper
    
    def track_function(self, func) -> None:
        """Decorator to track request count and latency, ignoring metric/tracing errors."""
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                tracer = getattr(self, "tracer", None)
                if tracer:
                    with tracer.start_as_current_span(func.__name__):
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            except Exception:
                self.logger.info("issue pushing trace to jaeger")
                result = func(*args, **kwargs)
            try:
                if hasattr(self, "LATENCY"):
                    elapsed = time.time() - start_time
                    self.LATENCY.observe(elapsed)
            except Exception:
                pass

            return result
        return wrapper

def get_hardware() -> str:
    """Get hardware available for inference.
    """
    hardware = "cpu"
    if torch.cuda.is_available():
        hardware = "cuda"
    else:
        if torch.backends.mps.is_available():
            hardware = "mps"
    return hardware

def get_root_dir() -> str:
    """Get root working directory of the project.
    """
    project_root = os.getenv(
        "PROJECT_ROOT", 
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        )
    return project_root
    

def get_model_dir(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
    """Get directory of the saved LLM model

    Args:
        model_name (str, optional): Model name on [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). 
        Defaults to "Qwen2.5-0.5B-Instruct".

    Returns:
        str: _description_
    """
    model_dir = os.path.join(get_root_dir(), "rag-pipeline/models/" + model_name)
    return model_dir

def get_doc_dir() -> str:
    """Get directory of the input PDF document.
    """
    doc_dir = os.path.join(get_root_dir(), "rag-pipeline/examples/example.pdf")
    return doc_dir
        
def secure_filename(filename: str) -> str:
    """Secure filename sanitizer
    """
    # Remove path components and special characters
    clean = re.sub(r'(?u)[^-\w.]', '', filename.split("/")[-1])
    # Prevent empty filenames and hidden files
    return clean.lstrip('.') or 'document'