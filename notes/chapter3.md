# Data Preprocessing
## Loading dataset
```
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```
```
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```
dataset 的内部结构
```
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

{
    "idx": 0,
    "label": 1,
    "sentence1": 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
    "sentence2": 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
}
```
> 一些模型再tokenize的时候会返回token_type_ids，用于区分第一句与第二句，一般会用于问答或者下一句预测。由于这种默认的处理方式，我们可以将两个sentences同时加载
```
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```
这种方法虽然有效，但有一个缺点是它返回的是一个字典（字典的键是 输入词id(input_ids) ， 注意力遮罩(attention_mask) 和 token类型ID(token_type_ids) ，字典的值是键所对应值的列表）。这意味着在转换过程中要有足够的内存来存储整个数据集才不会出错。不过来自🤗 Datasets 库中的数据集是以 Apache Arrow 格式存储在磁盘上的，因此你只需将接下来要用的数据加载在内存中，而不是加载整个数据集，这对内存容量的需求比较友好。

### Not loading data at once
使用 Dataset.map() 方法将数据保存为 dataset 格式，如果我们需要做更多的预处理而不仅仅是 tokenization 它还支持了一些额外的自定义的方法。 map() 方法的工作原理是使用一个函数处理数据集的每个元素。让我们定义一个对输入进行 tokenize 的函数：
```
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```
这样可以针对dataset中的每一条数据使用同样的function执行策略
`map` 的处理结果是向dataset中添加字段：“每一个sample都会被添加一样的处理后的字段结果”

### Dynamic padding
collate 方法会将sample组合成batch。这是一个可以在构建 DataLoader 时传递的一个参数，默认是一个将数据集转换为 PyTorch 张量并将它们拼接起来的函数（如果你的元素是列表、元组或字典，则会使用递归进行拼接）。需要注意的是，这不能够用来改变sample的长度

因此，在这个例子中，为了解决句子长度不统一的问题，我们必须定义一个 collate 函数，该函数会将每个 batch 句子填充到正确的长度。幸运的是，🤗transformer 库通过 DataCollatorWithPadding 为我们提供了这样一个函数。
```
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
> 为什么需要tokenizer作为入参？因为不同的tokenizer可能有不同的 [PAD] id，并且他们pad的方向位置也不同，所以需要tokenizer. 需要注意的是，给到collate方法的是已经tokenize之后的结果，而不是字符串.


