# Model Fintuning
## Data loading and processing
使用到之前的方法，用Tokenizer针对数据进行处理。如果想load数据集的一部份，可以`shuffle`之后使用其中一部份作为训练数据

## Train with Trainer class
使用TrainerArguments设置训练超参数
```python
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")
```
训练过程中没有提供Evaluate功能，需要`import evaluate`来load一些validation指标
```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
# 定义metric如何计算。因为每个结果都是logits格式输出，所以需要通过compute_metrics转换

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 每个epoch结束进行eval
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```

## Train with PyTorch
当使用pytorch训练的时候，需要将dataset类load到常用的Dataloader用于加载batch数据，定义优化器（如AdamW），手动eval，learning rate更新
```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# 正式训练
from tqdm.auto import tqdm
import evaluate
metric = evaluate.load("accuracy")

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()
```