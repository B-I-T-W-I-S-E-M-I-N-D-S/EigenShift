import torch
import numpy as np
import random
import pandas as pd
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name= 'meta-llama/LLaMA-2-7b-hf'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

token = '____YOUR_HF_TOKEN____'

model = AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token

W = np.array(model.lm_head.weight.data.cpu())  # W: (vocab_size, hidden_dim)

# Perform SVD
U, S, Vt = np.linalg.svd(W, full_matrices=False)

# Reconstruct the matrix
W_reconstructed = (U @ np.diag(S)) @ Vt

# Compute the Frobenius norm of the reconstruction error
reconstruction_loss = np.linalg.norm(W - W_reconstructed, ord='fro')

print(f"Frobenius norm (reconstruction loss): {reconstruction_loss:.6f}")