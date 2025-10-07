import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")

# Configuration
MODEL_NAME = 'meta-llama/Llama-2-7b-hf'
TOKEN = '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'
GENERATIONS_PER_PROMPT = 25
MAX_NEW_TOKENS = 20
DEVICE = 'cuda:0'

# CRITICAL: Process multiple prompts at once for GPU utilization
# With 4-bit quantization, we can increase batch size
PROMPT_BATCH_SIZE = 8  # Process 8 prompts simultaneously (4-bit uses less memory)
TOTAL_BATCH_SIZE = PROMPT_BATCH_SIZE * GENERATIONS_PER_PROMPT  # 200 sequences at once

# Load dataset
df = pd.read_csv('datasets/Real_toxic_prompts_toxicit_0.5_5000Sample_rs42.csv')

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
    MODEL_NAME,
    token=TOKEN,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # Important for batch generation

# ============================================
# Load classifier on GPU
# ============================================
print("Loading toxicity classifier...")
cls_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls.to(DEVICE)
cls.half()  # FP16 for faster inference
cls.eval()

# ============================================
# Optimized batch processing
# ============================================
all_generations = []
all_labels = []

print("Starting generation...")
prompts = df['prompt'].tolist()

# Process in batches of PROMPT_BATCH_SIZE
for i in tqdm(range(0, len(prompts), PROMPT_BATCH_SIZE), desc="Processing prompt batches"):
    batch_prompts = prompts[i:i + PROMPT_BATCH_SIZE]

    # Repeat each prompt GENERATIONS_PER_PROMPT times
    expanded_prompts = []
    for prompt in batch_prompts:
        expanded_prompts.extend([prompt] * GENERATIONS_PER_PROMPT)

    # Tokenize all at once
    inputs = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)

    # Generate with autocast
    with torch.no_grad(), autocast():
        output_tokens = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,  # Faster than beam search
            use_cache=True  # Enable KV cache
        )

    # Decode outputs
    decoded_outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    # Extract generated text (remove prompt)
    output_texts = []
    for j, (prompt, full_output) in enumerate(zip(expanded_prompts, decoded_outputs)):
        generated = full_output[len(prompt):].strip() if full_output.startswith(prompt) else full_output.split(prompt)[
            -1].strip()
        output_texts.append(generated)

    # Classify toxicity in large batches (faster)
    all_preds = []
    CLASSIFIER_BATCH_SIZE = 32

    for j in range(0, len(output_texts), CLASSIFIER_BATCH_SIZE):
        batch_texts = output_texts[j:j + CLASSIFIER_BATCH_SIZE]

        encoding = cls_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.no_grad(), autocast():
            outputs = cls(**encoding)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)

    # Reshape results back to per-prompt structure
    for j in range(len(batch_prompts)):
        start_idx = j * GENERATIONS_PER_PROMPT
        end_idx = start_idx + GENERATIONS_PER_PROMPT

        prompt_generations = output_texts[start_idx:end_idx]
        prompt_preds = all_preds[start_idx:end_idx]
        prompt_labels = ['yes' if pred == 1 else 'no' for pred in prompt_preds]

        all_generations.append(prompt_generations)
        all_labels.append(prompt_labels)

    # Clear cache less frequently (every 100 prompts)
    if (i + PROMPT_BATCH_SIZE) % 100 == 0:
        torch.cuda.empty_cache()

# ============================================
# Save results
# ============================================
df['generations'] = all_generations
df['toxicity'] = all_labels
df = df[['prompt', 'generations', 'toxicity']]

output_file = f"generations/wo-intervention/{MODEL_NAME.split('/')[-1]}_RTP_generations.json"
df.to_json(output_file, orient='records', indent=2)
print(f"\nResults saved to {output_file}")
print(f"Total prompts processed: {len(df)}")
