import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup


class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, files, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.tokens = []

        for file_path in files:
            with open(file_path, "r") as f:
                text = f.read()
            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)[
                "input_ids"].squeeze()
            self.tokens.append(tokens)

        self.tokens = torch.cat(self.tokens)

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size

        return self.tokens[start_idx:end_idx]


model_name = "gpt2"
epochs = 4
learning_rate = 5e-5
warmup_steps = 1e2
batch_size = 4
max_sequence_length = 128
# Replace with the path to your text corpus directory
train_dir = "train/"
# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load tokenizer functionality
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Load GPT-2 model configuration
config = GPT2Config.from_pretrained("gpt2")
# Create GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
# Get list of text files in train_dir
train_files = [os.path.join(train_dir, f)
               for f in os.listdir(train_dir) if f.endswith('.txt')]
# Initialize the CustomTextDataset with the tokenizer
train_dataset = CustomTextDataset(tokenizer, train_files, max_sequence_length)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

model.train()

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")

    for batch in train_dataloader:
        inputs = batch.to(device)
        labels = inputs.clone()

        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.item()}")
