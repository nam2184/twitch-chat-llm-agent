import html
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
from dotenv import load_dotenv
import os
from setup import LLMService, get_model_dir
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from util import get_root_dir

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

def twitch_dataset_to_pdf(output_pdf="twitch_chat_sample.pdf", max_size_mb=40):
    """
    Save up to ~40 MB of Twitch chat messages from HuggingFace dataset to a PDF file.
    """

    dataset = load_dataset("lparkourer10/twitch_chat")
    messages = dataset["train"]["Message"]

    # --- Estimate how many samples to take ---
    # We'll sample messages until roughly 40MB of text is reached
    max_bytes = max_size_mb * 1024 * 1024
    total_bytes = 0
    selected_messages = []

    for msg in messages:
        if not isinstance(msg, str) or not msg.strip():
            continue
        msg_bytes = len(msg.encode("utf-8"))
        if total_bytes + msg_bytes > max_bytes:
            break
        selected_messages.append(msg.strip())
        total_bytes += msg_bytes

    print(f"Selected {len(selected_messages)} messages (~{total_bytes/1024/1024:.2f} MB)")

    # --- Create PDF ---
    os.makedirs(get_root_dir() + f"/tiny-rag-llm-agent/examples", exist_ok=True)
    doc = SimpleDocTemplate(get_root_dir() + f"/tiny-rag-llm-agent/examples/{output_pdf}", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Twitch Chat Sample Dataset</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Total messages: {len(selected_messages)}", styles["Normal"]))
    story.append(Paragraph(f"Approx size: {total_bytes/1024/1024:.2f} MB", styles["Normal"]))
    story.append(Spacer(1, 12))

    for i, msg in enumerate(selected_messages, 1):
        safe_msg = html.escape(msg)  # safely escape <, >, &, etc.
        try:
            story.append(Paragraph(f"<b>{i}.</b> {safe_msg}", styles["Normal"]))
        except Exception as e:
            print(f"Skipped malformed message {i}: {e}")
        story.append(Spacer(1, 6))

    doc.build(story)
    print(f"PDF created: {output_pdf}")

if __name__ == "__main__":
    #twitch_finetune_qwen(output_dir="./twitchtuned_qwen", epochs=2, lr=2e-5)
    #upload_to_hf("./twitchtuned_qwen", f"{os.getenv('HF_USERNAME')}/qwen2.5B-twitchfinetuned")
    twitch_dataset_to_pdf(max_size_mb=1)