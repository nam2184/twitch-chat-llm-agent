import os
import logging
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from util import Logger, RAGMetrics, get_hardware, get_model_dir

logger = Logger(name="LLM")

class LLM:
    def __init__(self):
        self.logger = Logger(name="LLM", id=self.id())
        self.metrics = RAGMetrics()
        self.load_model = self.metrics.track_function(self.load_model)

    def load_model(self, model_name: str, local_dir: str) -> any:
        """Setup local LLM model for inference.

        Args:
            model_name (str): Path to download the model from Hugging Face Hub.
            local_dir (str): Local directory to save the model.
        """
        # Ensure the directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            logger.info(f"Model not found locally, downloading to {local_dir}...")
            snapshot_download(repo_id=model_name, local_dir=local_dir)
            logger.info("Model downloaded!")
        else:
            logger.info(f"Model already exists in {local_dir}, skipping download.")
        
        # Get hardware
        hardware = get_hardware()
        
        logger.info(f"Loading model from {local_dir} on {hardware}...")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=local_dir,
            device_map=hardware
            )    

        return model
    
if __name__ == "__main__":
    # We use Qwen2.5-0.5B-Instruct as our local LLM model (~1GB).
    # Link to download the model:
    # https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e
    model = LLM().load_model(
        model_name="Qwen/Qwen2.5-0.5B-Instruct", 
        local_dir=get_model_dir()
    )
    print("model:", model)
    print("Model loaded successfully!")