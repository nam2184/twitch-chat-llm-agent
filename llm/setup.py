import os
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from util import Logger, RAGMetrics, get_hardware, get_model_dir, trace
from transformers import AutoTokenizer, pipeline
from transformers.modeling_utils import PreTrainedModel
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.tools import Tool
from langchain.agents import create_agent
from transformers import AutoTokenizer, pipeline
from pipeline import RAG
from fastapi import UploadFile, File
import asyncio

class LLMService:
    def __init__(self, logger : Logger = None, metrics : RAGMetrics = None):
        self.logger = logger or Logger(name="LLM", id=id(self))
        self.metrics = metrics or RAGMetrics()
        self.load_model = self.metrics.track_function(self.load_model)
        self.model = None
        self.RAG = RAG(self.logger, self.metrics)
        self.Agents = {} 
    
    async def setup_pipeline_for_user(self, user_id: str, file: UploadFile = None):
        """Setup a RAG chain for a specific user.

        Args:
            user_id (str): User ID.
            file_path (str): File path to the PDF file.
        """
        if user_id not in self.Agents:
            self.Agents[user_id] = await self.setup_pipeline(
                local_dir=get_model_dir(),
                file=file,
                model=self.model,
                user_id=user_id
            )
            self.logger.info(f"QA chain set up for user {user_id}.")
        else:
            self.logger.info(f"QA chain already exists for user {user_id}, skipping setup.")

    def load_model(self, model_name: str, local_dir: str) -> any:
        """Setup local LLM model for inference.

        Args:
            model_name (str): Path to download the model from Hugging Face Hub.
            local_dir (str): Local directory to save the model.
        """
        # Ensure the directory exists
        os.makedirs(local_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            self.logger.info(f"Model not found locally, downloading to {local_dir}...")
            snapshot_download(repo_id=model_name, local_dir=local_dir)
            self.logger.info("Model downloaded!")
        else:
            self.logger.info(f"Model already exists in {local_dir}, skipping download.")
        
        # Get hardware
        hardware = get_hardware()
        
        self.logger.info(f"Loading model from {local_dir} on {hardware}...")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=local_dir,
            device_map=hardware
            )   

        return model
    
    async def setup_pipeline(self, local_dir: str, file: UploadFile = None, model: PreTrainedModel = None, user_id: str = None) -> any:
        """Setup a QA chain with RAG model.

        Args:
            local_dir (str): Directory of the local LLM model.
            file_path (str): File path to the PDF file.
            model (PreTrainedModel): Pre-loaded locam LLM model.

        Returns:
            An Agent object.
        """
        with self.metrics.tracer.start_as_current_span("setup_pipeline") as setup_pipeline:
            with self.metrics.tracer.start_as_current_span("load_tokenizer", links=[trace.Link(setup_pipeline.get_span_context())]):
                tokenizer = AutoTokenizer.from_pretrained(local_dir)
            
            if model is None:
                model = self.load_model(
                    model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                    local_dir=get_model_dir()
                )
            
            with self.metrics.tracer.start_as_current_span("prepare_retriever", links=[trace.Link(setup_pipeline.get_span_context())]):
                retriever = await self.RAG.prepare_retriever_file(file=file, user_id=user_id)
            
            # Setup a RAG pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                repetition_penalty=1.2,
                max_new_tokens=500
            )
            
            # Wrap the HuggingFace pipeline in a LangChain object
            local_llm = HuggingFacePipeline(pipeline=pipe)
            chat_model = ChatHuggingFace(llm=local_llm) 
            # Define the prompt template
            prompt_template = """Answer based on context:\n{context}\nQuestion: {question}\nAnswer:"""
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt_template
            )
            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve user pdfs",
                "Search and return information user pdfs",
                document_prompt=prompt,
            )

            return create_agent(
                tools=[retriever_tool],
                model=chat_model,
                name="twitch-llm-agent",
            )


if __name__ == "__main__":
    # We use Qwen2.5-0.5B-Instruct as our local LLM model (~1GB).
    # Link to download the model:
    # https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e
    llm = LLMService()
    model = llm.load_model(
        model_name="Qwen/Qwen2.5-0.5B-Instruct", 
        local_dir=get_model_dir()
    )
    agent = asyncio.run(llm.setup_pipeline(
        local_dir=get_model_dir(),
        model=model
    ))
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Explain machine learning"}]},
        context={"user_role": "expert"}
    )
    print(result)
    print("model:", model)
    print("Model loaded successfully!")