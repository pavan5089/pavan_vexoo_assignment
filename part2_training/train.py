# part2_training/train.py

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# 1. Load Dataset

dataset = load_dataset("gsm8k", "main")

train_data = dataset["train"].select(range(3000))
test_data = dataset["test"].select(range(1000))


# 2. Tokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

def tokenize(example):
    return tokenizer(example["question"], truncation=True, padding="max_length")

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)


# 3. Model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


# 4. Training Loop (Simple)

model.train()

for i, sample in enumerate(train_data):
    inputs = torch.tensor(sample["input_ids"]).unsqueeze(0)
    outputs = model(inputs, labels=inputs)

    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item()}")

    if i > 500:  # limit for demo
        break

# 5. Evaluation

model.eval()
correct = 0

for sample in test_data.select(range(100)):
    inputs = torch.tensor(sample["input_ids"]).unsqueeze(0)
    output = model.generate(inputs, max_length=50)

    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    actual = sample["answer"]

    if actual.strip() in pred:
        correct += 1

accuracy = correct / 100
print("Accuracy:", accuracy)