from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
from dotenv import load_dotenv
import os
from setup import LLMService, get_model_dir

load_dotenv()

def twitch_finetune_qwen(output_dir="./twitchtuned_qwen", epochs=2, lr=2e-5):
    """
    Fine-tune Qwen2.5-0.5B on a small dataset.

    Args:
        model: Pre-loaded Qwen model (AutoModelForCausalLM)
        tokenizer: Matching tokenizer (AutoTokenizer)
        data: list of dicts, each like {"input": "question", "output": "answer"}
        output_dir: where to save fine-tuned weights
        epochs: number of training epochs
        lr: learning rate

    Returns:
        Trained model and tokenizer
    """
    llm = LLMService()
    model = llm.load_model(
        model_name="Qwen/Qwen2.5-0.5B-Instruct", 
        local_dir=get_model_dir()
    )
    tokenizer = AutoTokenizer.from_pretrained(get_model_dir())
    
    dataset = load_dataset("lparkourer10/twitch_chat")["train"]
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),  # Use bf16 if available
        report_to="none",
        optim="adamw_torch",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer

def upload_to_hf(model_dir, repo_name, private=True):
    """
    Upload a fine-tuned model directory to Hugging Face Hub.
    """
    print(f"Uploading model from {model_dir} to Hugging Face repo {repo_name}...")
    from huggingface_hub import HfApi
    api = HfApi()

    # Create (or reuse) the repo
    api.create_repo(repo_name, exist_ok=True, private=private)

    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_name,
        path_in_repo=".",
    )

    print(f"Uploaded {model_dir} → https://huggingface.co/{repo_name}")

twitch_finetune_qwen(output_dir="./twitchtuned_qwen", epochs=2, lr=2e-5)
upload_to_hf("./twitchtuned_qwen", f"{os.getenv('HF_USERNAME')}/qwen2.5B-twitchfinetuned")

