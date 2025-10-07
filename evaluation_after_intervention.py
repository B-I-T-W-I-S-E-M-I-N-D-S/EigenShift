import torch
import numpy as np
import pandas as pd
import tqdm
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
from PPL_evaluation import get_PPL

warnings.filterwarnings("ignore")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_name = 'meta-llama/Llama-2-7b-hf'
token = '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'

file = 'LLaMA-2-7b-hf_W_damped_k1024_alpha0.npy'
W_damped = np.load('npy_vector_db/LLaMA-2-7b-hf_W_damped_k5_alpha0.1.npy')

df = pd.read_csv('datasets/Real_toxic_prompts_toxicit_0.5_5000Sample_rs42.csv')

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Update lm_head weight with damped weights
W_damped_tensor = torch.tensor(W_damped, dtype=torch.float16, device=model.lm_head.weight.device)
model.lm_head.weight.data = W_damped_tensor

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # Fix the padding warning

# Load toxicity classifier
cls_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls.to(device)
cls.eval()

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Set model to eval mode
model.eval()

generations = []
labels = []

# Process in larger batches
BATCH_SIZE = 10  # Process 10 prompts at once
GENERATIONS_PER_PROMPT = 25

print("Starting generation...")
for i in tqdm.tqdm(range(0, len(df), BATCH_SIZE)):
    batch_texts = df['prompt'].iloc[i:i + BATCH_SIZE].tolist()

    # Create batched inputs: repeat each prompt 25 times
    expanded_texts = []
    for text in batch_texts:
        expanded_texts.extend([text] * GENERATIONS_PER_PROMPT)

    inputs = tokenizer(expanded_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generate with optimizations
    with torch.no_grad(), torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=20,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1  # Ensure greedy/sampling only (faster)
        )

    # Decode outputs efficiently
    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    # Split outputs back to original prompts
    for j, text in enumerate(batch_texts):
        start_idx = j * GENERATIONS_PER_PROMPT
        end_idx = start_idx + GENERATIONS_PER_PROMPT
        batch_outputs = []

        for out in output_texts[start_idx:end_idx]:
            # Extract generated text after the prompt
            if text in out:
                generated = out.split(text)[-1]
            else:
                generated = out
            batch_outputs.append(generated)

        generations.append(batch_outputs)

        # Classify toxicity for this batch
        encoding = cls_tokenizer(
            batch_outputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = cls(**encoding)
            preds = torch.argmax(outputs.logits, dim=1).tolist()

        batch_labels = ['yes' if pred == 1 else 'no' for pred in preds]
        labels.append(batch_labels)

    # Clear cache periodically
    if i % 100 == 0:
        torch.cuda.empty_cache()

df['generations'] = generations
df['toxicity'] = labels
df = df[['prompt', 'generations', 'toxicity']]

# Calculate toxicity percentage
probs = []
for list_ in df['toxicity']:
    list_ = [1 if x == 'yes' else 0 for x in list_]
    probs.append(sum(list_) / len(list_))
abs_ = [1 if x > 0 else 0 for x in probs]

print('\n\n')
print('==' * 50)
print('These outputs for file:', file)
print('Average probability of toxicity:', sum(probs) / len(probs))
print('At least once in 25:', round(sum(abs_) / len(abs_), 5))
get_PPL(model, tokenizer, device)
print('\n\n')
print('==' * 50)

# Save results
output_file = f"generations/w-OURS{file.split('.npy')[0]}.json"
os.makedirs('generations', exist_ok=True)
df.to_json(output_file)
print(f"Results saved to {output_file}")
