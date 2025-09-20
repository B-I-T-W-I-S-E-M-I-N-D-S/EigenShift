import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pathlib
from tqdm import tqdm
import argparse
import typing as t
import numpy as np

# Dataset
class WikipediaDataset(Dataset):
    def __init__(self, csv_path, num_sentences=5000):
        df = pd.read_csv(csv_path)
        self.sentences = df["text"].tolist()[:num_sentences]

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)


# Perplexity function
def perplexity_batch(sentences, tokenizer, model, device="cuda", max_length=128):
    model.eval()
    with torch.no_grad():
        tok_out = tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids = tok_out.input_ids
        attention_mask = tok_out.attention_mask
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.to(torch.float64)

        ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = ce(logits[:, :-1, :].permute(0, 2, 1), labels[:, 1:])
        loss_mask = attention_mask[:, 1:].to(torch.float64)

        sent_ppl = torch.exp(torch.sum(loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1))
        return sent_ppl


# Measure perplexity over dataset
def measure_perplexity(model, tokenizer, dataloader, device):
    all_ppls = []
    for batch in tqdm(dataloader):
        ppl = perplexity_batch(batch, tokenizer, model, device=device)
        all_ppls.append(ppl)

    all_ppls = torch.cat(all_ppls)
    return all_ppls.mean().item(), all_ppls.std().item()


def get_PPL(model,tokenizer, device):
    class Args:
        def __init__(self):
            self.model_path = model
            self.dataset_path = '/datasets/wikipedia_sentences.csv'
            self.device = device
            self.num_sentences = 5000
            self.batch_size = 4
    args = Args()
    device = args.device
    # Load model
    tokenizer.pad_token = tokenizer.eos_token
    # Load data
    dataset = WikipediaDataset(args.dataset_path, num_sentences=args.num_sentences)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Measure perplexity
    mean_ppl, std_ppl = measure_perplexity(model, tokenizer, dataloader, device)
    print(f"Perplexity (mean): {mean_ppl:.2f}")
    print(f"Perplexity (std): {std_ppl:.2f}")