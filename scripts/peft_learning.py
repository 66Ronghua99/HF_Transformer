from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
import torch
from tqdm.auto import tqdm
import argparse
import os
import math

# create LoRA configuration object
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    target_modules=["c_proj", "c_attn"],
    lora_dropout=0.1 # dropout of LoRA layers
)


model = AutoModelForCausalLM.from_pretrained("distilgpt2")
# 训练时关闭缓存以支持梯度检查点
if hasattr(model, "config"):
    model.config.use_cache = False

from datasets import load_dataset
from transformers import AutoTokenizer

# 解析命令行参数（明确限制使用的 GPU）
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use among visible devices; ignored if no CUDA")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
raw_dataset = load_dataset("imdb")

train_dataset = raw_dataset["train"]
eval_dataset = raw_dataset["test"]

# 如果模型没有 pad token，需要手动设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # 将文本和标签组合成一个完整的序列
    # 格式化为 "文本: {评论} 标签: {标签}"
    # 这一步对于因果语言模型（Causal LM）的微调至关重要，因为它需要学习生成完整的序列
    text_with_labels = [
        f"Review: {text} Label: {label_name}"
        for text, label_name in zip(examples['text'], [train_dataset.features['label'].int2str(label) for label in examples['label']])
    ]
    
    # 对组合后的文本进行分词
    # truncation=True 截断过长的序列
    # padding=True 将短序列填充到统一长度
    # return_tensors="pt" 返回 PyTorch 张量
    return tokenizer(text_with_labels, truncation=True, padding=True)

def tokenize_eval_function(examples):
    text_with_labels = [
        f"Review: {text}"
        for text, label_name in zip(examples['text'], [eval_dataset.features['label'].int2str(label) for label in examples['label']])
    ]
    
    # 对组合后的文本进行分词
    # truncation=True 截断过长的序列
    # padding=True 将短序列填充到统一长度
    # return_tensors="pt" 返回 PyTorch 张量
    return tokenizer(text_with_labels, truncation=True, padding=True)

# 应用分词函数到整个数据集（标签由 collator 动态生成）
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

output_dir = "models/lora_finetuning"
learning_rate = 2e-4
train_batch_size = 4
eval_batch_size = 4
num_train_epochs = 2
weight_decay = 0.01
warmup_ratio = 0.03
gradient_accumulation_steps = 4
max_grad_norm = 1.0

# 建立 LoRA 适配器
model = get_peft_model(model, lora_config)

# 启用梯度检查点可节省显存（若模型支持）
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    # 一些模型在开启梯度检查点时需要启用输入的梯度
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

# 明确限制到指定 GPU（带边界检查）；若无 CUDA 则回退到 CPU
if torch.cuda.is_available():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    # 在多用户环境下，CUDA_VISIBLE_DEVICES 可能将物理 GPU 映射为连续的 0..N-1
    num_visible = torch.cuda.device_count()
    if num_visible == 0:
        device = torch.device("cpu")
    else:
        if args.gpu < 0 or args.gpu >= num_visible:
            raise ValueError(f"Requested GPU index {args.gpu} out of range [0, {num_visible - 1}] for visible devices {visible_devices or 'ALL'}")
        try:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
        except Exception as e:
            # 某些环境下 torch 在惰性初始化时设备总数与预检查不一致（例如显示有8卡但内部枚举为7）
            # 建议使用环境变量固定物理卡映射： CUDA_VISIBLE_DEVICES=<physical_id> python ... --gpu 0
            msg = (
                f"Failed to set cuda device {args.gpu}: {e}\n"
                f"Tip: pin the physical GPU via CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES={args.gpu} python peft_learning.py --gpu 0"
            )
            raise RuntimeError(msg) from e
else:
    device = torch.device("cpu")

# 数据整理器：针对 Causal LM（非 MLM）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataloader = DataLoader(
    tokenized_train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=train_batch_size,
)

eval_dataloader = DataLoader(
    tokenized_eval_dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=eval_batch_size,
)

# 仅优化可训练参数（LoRA 参数）
trainable_params = [p for p in model.parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise ValueError("No trainable parameters found. Check LoRA target_modules to ensure adapters are attached.")
optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

# 估算总步数以建立调度器
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = num_train_epochs * num_update_steps_per_epoch
num_warmup_steps = int(max_train_steps * warmup_ratio)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
)

model = model.to(device)
# 为了明确限制，只使用指定的单卡，不做 DataParallel 包裹

# 训练循环
for epoch in range(num_train_epochs):
    model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_dataloader)
    progress_bar.set_description(f"Epoch {epoch+1}/{num_train_epochs}")

    optimizer.zero_grad()
    for step, batch in enumerate(progress_bar):
        # 将 batch 移动到设备
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        # 回退计算（罕见情况下）
        if not loss.requires_grad:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_train_loss += loss.detach().float()
        if (step + 1) % 20 == 0:
            progress_bar.set_postfix({"loss": f"{(total_train_loss / (step + 1)).item():.4f}"})

    # 评估
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            eval_losses.append(loss.detach().float().cpu())

    if len(eval_losses) > 0:
        mean_eval_loss = torch.stack(eval_losses).mean().item()
        perplexity = math.exp(mean_eval_loss)
        print(f"Epoch {epoch+1}: eval_loss={mean_eval_loss:.4f}, ppl={perplexity:.2f}")

# 仅在主进程保存
# 保存（兼容 DataParallel 包裹）
to_save = model.module if hasattr(model, "module") else model
to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Saved LoRA adapter and tokenizer to {output_dir}")