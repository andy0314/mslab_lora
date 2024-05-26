from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# 加載訓練好的模型和 tokenizer
model = GPT2LMHeadModel.from_pretrained('./exp_5')
tokenizer = GPT2Tokenizer.from_pretrained('./exp_5')

tokenizer.pad_token = tokenizer.eos_token

num = 1000
size = 10

ip =      "0123456789"
mapping = "abcdefghij"

correct = 0
turn = 0

data = []

for i in range(num):
    data_i = ""
    data_o = ""
    idxs = np.random.randint(0, len(ip), size)
    for x in idxs:
        data_i += ip[x]
        data_o += mapping[x]
    input_prompt = f"Q:{data_i} A:"
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=20,  # 生成文本的最大長度
        num_return_sequences=1,  # 生成的文本數量
        no_repeat_ngram_size=2,  # 防止重複的 n-gram
        pad_token_id = tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if len(generated_text) > 25:
        generated_text = generated_text[:25]
    result = generated_text[-10:]
    for x in range(10):
        if data_o[x] == result[x]:
            correct += 1
        turn += 1
    data += f"{generated_text}\n"
    print(i, data_o, result)

f = open("result.txt", "x")
for d in data:
    f.write(d)
f.close()

print(correct, turn)
