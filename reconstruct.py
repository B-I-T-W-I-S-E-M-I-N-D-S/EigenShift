import torch
import numpy as np
import random
import pandas as pd
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import math

captured_hidden_states = []


def hook_fn(module, input, output):
    captured_hidden_states.append(input[0].detach().cpu()[0][0])


def get_output(text, input_ids=None, input_text='False', max_new_tokens=20):
    output_tupple = []
    hook = model.lm_head.register_forward_hook(hook_fn)

    if input_ids is None:
        inputs_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
    else:
        inputs_ids = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = model.generate(inputs=inputs_ids, max_new_tokens=max_new_tokens, temperature=1.0, do_sample=False,
                                 top_p=None)

    outputs = [int(x) for x in outputs[0]]
    inputs = [int(x) for x in inputs_ids[0]]
    outputs = outputs[len(inputs):]
    hook.remove()
    if input_text == 'True':
        for x in zip(outputs):
            output_tupple.append((x, tokenizer.decode(x)))
    return captured_hidden_states, output_tupple


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_name = 'meta-llama/Llama-2-7b-hf'

df = pd.read_csv(f"generations/wo-intervention/LLaMA-2-7b-hf_RTP_generations_roberta_toxic_words_extraction.csv")

token = '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'

# ============================================
# Load model with 4-bit quantization
# ============================================
print("Loading model in 4-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
# ============================================

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token

logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    non_toxic_vectors = np.load(f"npy_vector_db/non_toxic_vectors_{model_name.split('/')[-1]}.npy")
    print('Loaded non-toxic vectors from file')
    print(non_toxic_vectors.shape)
except FileNotFoundError:
    non_toxic_vectors = []
    non_toxic_df = df[df['label'] == 'no']
    non_toxic_df.reset_index(inplace=True)

    for prompt in tqdm.tqdm(non_toxic_df['prompt']):
        captured_hidden_states = []
        hidden_states, output_tupple = get_output(prompt, max_new_tokens=1)
        non_toxic_vectors.append(hidden_states[0].numpy())
    non_toxic_vectors = np.array(non_toxic_vectors)
    np.save(f"npy_vector_db/non_toxic_vectors_{model_name.split('/')[-1]}.npy", non_toxic_vectors)
# 100%|██████████| 4349/4349 [03:08<00:00, 23.03it/s]

# ============================================
# Get weight matrix (dequantize for 4-bit model)
# ============================================
W = model.lm_head.weight.data.float().cpu().numpy()  # shape (32000, 4096)
# ============================================

logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    toxic_vectors = np.load(f"npy_vector_db/toxic_vectors_{model_name.split('/')[-1]}.npy")
    print('Loaded toxic vectors from file')
    print(toxic_vectors.shape)
except FileNotFoundError:
    toxic_vectors = []
    toxic_df = df[df['label'] == 'yes']
    toxic_df.reset_index(inplace=True)

    for N, (prompt, generation, toxic_word) in enumerate(
            tqdm.tqdm(zip(toxic_df['prompt'], toxic_df['generation'], toxic_df['toxic_words']),
                      desc="Processing output tuples", total=len(toxic_df))):
        try:
            toxic_word = eval(toxic_word)[0]
            captured_hidden_states = []
            hidden_states, output_tuple = get_output(prompt, max_new_tokens=20, input_text='True')

            for n, (x, y) in enumerate(output_tuple):
                if y in toxic_word:
                    toxic_index = n
                    predicted_token = tokenizer.decode(np.argmax(np.dot(np.array(hidden_states[toxic_index]), W.T)))
                    if predicted_token == y:
                        toxic_vector = hidden_states[toxic_index]
                        toxic_vectors.append(toxic_vector)
                    else:
                        print('Failed at:', N)
                    break
        except Exception as e:
            print('Error:', e, 'at:', N, 'toxic_word:', toxic_word)

        # break
    toxic_vectors = np.array(toxic_vectors)
    np.save(f"npy_vector_db/toxic_vectors_{model_name.split('/')[-1]}.npy", toxic_vectors)
# Processing output tuples: 100%|██████████| 651/651 [05:42<00:00,  1.90it/s]

before = [tokenizer.decode(np.argmax(np.dot(np.array(toxic_vectors[i]), W.T))) for i in range(50)]
print('Toxic vocabulary before intervention:\n', before)

# Step 1: Assume W is your lm_head weight matrix
U, S, Vt = np.linalg.svd(W, full_matrices=False)  # W = U Σ V^T

print('U , S, Vt:\n', U.shape, S.shape, Vt.shape)

A = np.array(Vt.T)  # ensure it's an ndarray

non_toxic_vecs = np.stack(non_toxic_vectors)  # shape: (5, 4096)
toxic_vecs = np.stack(toxic_vectors)  # shape: (5, 4096)

# Project all hidden states onto all eigenvectors
non_toxic_activations = A @ non_toxic_vecs.T
toxic_activations = A @ toxic_vecs.T

# Compute mean activation for each eigenvector
mean_non_toxic = np.mean(non_toxic_activations, axis=1)  # shape: (4096,)
mean_toxic = np.mean(toxic_activations, axis=1)  # shape: (4096,)

# Compute delta
delta = mean_toxic - mean_non_toxic  # shape: (4096,)

percentile = 0.999
top_k = math.ceil(4096 * (1 - percentile))

top_indices = np.argsort(delta)[top_k:][::-1]  # most positive = more toxic

print("Top-k toxicity-related eigenvector indices:", top_indices)
print("Delta values:", delta[top_indices])
# Intervention:
alpha = 0.1  # Experiment on alpha
for i in top_indices[:top_k]:
    S[i] *= alpha
W_damped = (U * S) @ Vt  # Element-wise multiply each column of U by new S

now = [tokenizer.decode(np.argmax(np.dot(np.array(toxic_vectors[i]), W_damped.T))) for i in range(50)]
for x, y in zip(before, now):
    print(x, '||', y)

# ============================================
# CRITICAL FIX: Convert back to float16 and proper device placement
# ============================================
W_damped_tensor = torch.tensor(W_damped, dtype=torch.float16)

# For quantized models, you need to handle this carefully
# The weight might be a quantized parameter, so we need to replace it properly
if hasattr(model.lm_head.weight, 'quant_state'):
    # This is a quantized weight - need special handling
    print("Warning: Modifying quantized weights. Results may be suboptimal.")
    print("Consider using full precision model for intervention.")
    # Convert to the expected format
    model.lm_head.weight = torch.nn.Parameter(W_damped_tensor.to(device))
else:
    # Regular weight assignment
    model.lm_head.weight.data = W_damped_tensor.to(device)

# Ensure model is on correct device
model = model.to(device)
# ============================================

toxic_df = df[df['label'] == 'yes']
toxic_df.reset_index(inplace=True)
toxic_df = toxic_df[0:]
for N, (prompt, generation, toxic_word) in enumerate(
        tqdm.tqdm(zip(toxic_df['prompt'], toxic_df['generation'], toxic_df['toxic_words']),
                  desc="Processing output tuples", total=len(toxic_df))):
    toxic_word = eval(toxic_word)[0]
    print('prompt:', prompt)
    print('expected:', generation)
    print('\nActual toxic:', toxic_word)
    captured_hidden_states = []
    hidden_states, output_tuple = get_output(prompt, max_new_tokens=20, input_text='True')
    break
np.save(f"npy_vector_db/{model_name.split('/')[-1]}_W_damped_k{top_k}_alpha{alpha}.npy", W_damped)
