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
    Fine-tune Qwen2.5-0.5B on Twitch chat messages dataset.
    """

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset("lparkourer10/twitch_chat")
    print("Dataset columns:", dataset["train"].column_names)

    def tokenize_fn(examples):
        # Clean message text
        texts = [m if m and isinstance(m, str) else "" for m in examples["Message"]]
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        # labels = input_ids copy for causal LM
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,  # remove old 'Message'
        desc="Tokenizing dataset",
    )

    # --- Training setup ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
        optim="adamw_torch",
        remove_unused_columns=False,  # keep all inputs for Trainer
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fine-tuning complete. Model saved to {output_dir}")
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

