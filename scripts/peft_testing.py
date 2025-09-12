import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

base_model_name = "distilgpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# GPT-2 family has no pad token by default; align pad_token with eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
raw_dataset = load_dataset("imdb")
eval_dataset = raw_dataset["test"]
def tokenize_eval_function(examples):
    texts = [f"Review: {text}" for text in examples['text']]
    return tokenizer(texts, truncation=True, padding=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_eval_function, batched=True, remove_columns=eval_dataset.column_names)

from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
eval_dataloader = DataLoader(
    tokenized_eval_dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

def compute_token_level_avg_loss(model: AutoModelForCausalLM) -> float:
    from tqdm.auto import tqdm
    model.eval()
    model.to(device)
    total_loss_times_tokens = 0.0
    total_valid_tokens = 0
    progress_bar = tqdm(eval_dataloader)
    progress_bar.set_description(f"Evaluating")
    with torch.inference_mode():
        for item, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            labels = batch.get("labels")
            if labels is None:
                continue
            valid_tokens = (labels != -100).sum().item()
            if valid_tokens == 0:
                continue
            total_loss_times_tokens += loss.item() * valid_tokens
            total_valid_tokens += valid_tokens
    return float(total_loss_times_tokens / max(total_valid_tokens, 1))

# Evaluate base model (without adapter)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_eval_loss = compute_token_level_avg_loss(base_model)

# Evaluate adapter-enabled model (with LoRA)
peft_base = AutoModelForCausalLM.from_pretrained(base_model_name)
peft_model = PeftModel.from_pretrained(peft_base, "models/lora_finetuning")
peft_model.config.pad_token_id = tokenizer.pad_token_id
peft_eval_loss = compute_token_level_avg_loss(peft_model)

print(f"Eval token-level avg loss | base(no adapter): {base_eval_loss:.4f}")
print(f"Eval token-level avg loss | with adapter   : {peft_eval_loss:.4f}")