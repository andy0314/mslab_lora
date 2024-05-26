from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer

# 加載自定義數據集
dataset = load_dataset('text', data_files={'train': 'train.txt'})

# 使用 GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 加載 GPT-2 模型
model = GPT2LMHeadModel.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

# 令牌化數據集
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 使用 DataCollatorForLanguageModeling 來處理數據
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 訓練參數設置
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# 創建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
)

# 開始訓練
trainer.train()

# 保存模型和 tokenizer
model.save_pretrained('./exp_5')
tokenizer.save_pretrained('./exp_5')
