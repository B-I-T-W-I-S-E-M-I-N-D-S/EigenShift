import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification, BitsAndBytesConfig
from captum.attr import Saliency
import warnings
import gc

warnings.filterwarnings("ignore")

# Configuration
MODEL_NAME = 'meta-llama/LLaMA-2-7b-hf'
TOKEN = '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'
DEVICE = 'cuda:0'
BATCH_SIZE = 4  # Process multiple prompts at once
GEN_BATCH_SIZE = 1  # For generation (memory intensive)

# Enable memory optimization
torch.cuda.empty_cache()
gc.collect()

# Load dataset
df = pd.read_csv('llama-2-7b_RTP_5000_generations.csv')

# ============================================
# Load model with 4-bit quantization
# ============================================
print("Loading LLaMA model in 4-bit quantization...")
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

# ============================================
# Load Roberta Toxicity Classifier (same GPU, FP16)
# ============================================
print("Loading toxicity classifier...")
cls_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls_model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls_model.half()  # FP16 to save memory
cls_model.eval()
cls_model.to(DEVICE)


# ============================================
# Wrapper for Saliency
# ============================================
class RobertaWrapper(torch.nn.Module):
    def __init__(self, model, attention_mask):
        super().__init__()
        self.model = model
        self.attention_mask = attention_mask

    def forward(self, inputs_embeds):
        return self.model.roberta(
            attention_mask=self.attention_mask,
            inputs_embeds=inputs_embeds
        ).last_hidden_state[:, 0, :] @ self.model.classifier.out_proj.weight.T + self.model.classifier.out_proj.bias


# ============================================
# Optimized toxic token extraction
# ============================================
def get_toxic_tokens(text, threshold=0.1):
    """Extract toxic tokens with memory-efficient processing"""
    try:
        # Tokenize on GPU
        encoding = cls_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128  # Increased from 30 for better context
        ).to(DEVICE)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Get embeddings (FP16)
        with torch.cuda.amp.autocast():
            input_embed = cls_model.roberta.embeddings(input_ids)

        input_embed = input_embed.detach().float()  # Convert to FP32 for gradients
        input_embed.requires_grad = True

        # Create wrapper and compute saliency
        wrapper = RobertaWrapper(cls_model, attention_mask)
        saliency = Saliency(wrapper)

        grads = saliency.attribute(inputs=input_embed, target=1, abs=True)
        grads_sum = grads.sum(dim=-1).squeeze(0)

        # Get tokens
        tokens = cls_tokenizer.convert_ids_to_tokens(input_ids[0])

        # Find top toxic tokens
        top_tokens = []
        topk = torch.topk(grads_sum, k=min(5, len(tokens)))  # Get top 5 or fewer

        for i in topk.indices:
            if grads_sum[i] > threshold:
                token = tokens[i]
                token_str = cls_tokenizer.convert_tokens_to_string([token]).strip()
                if token_str and token_str in text and token_str not in ['<s>', '</s>', '<pad>']:
                    top_tokens.append(token_str)

        # Clean up
        del input_embed, grads, wrapper, saliency

        return top_tokens

    except Exception as e:
        print(f"Error in get_toxic_tokens: {e}")
        return []


# ============================================
# Main processing loop
# ============================================
generations = []
labels = []
toxic_words_list = []

print("Starting generation and toxic word extraction...")

for i in tqdm(range(0, len(df), GEN_BATCH_SIZE), desc="Processing prompts"):
    batch_prompts = df['prompt'][i:i + GEN_BATCH_SIZE].tolist()

    # Generate text
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode outputs
    output_texts = [
        tokenizer.decode(output, skip_special_tokens=True).split(prompt)[-1].strip()
        for output, prompt in zip(output_tokens, batch_prompts)
    ]

    generations.extend(output_texts)

    # Free generation memory before classification
    del inputs, output_tokens
    torch.cuda.empty_cache()

    # Classify toxicity
    encoding = cls_tokenizer(
        output_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = cls_model(**encoding)
        preds = torch.argmax(outputs.logits, dim=1).tolist()

    batch_labels = ['yes' if p == 1 else 'no' for p in preds]
    labels.extend(batch_labels)

    # Extract toxic words only for toxic generations
    for text, label in zip(output_texts, batch_labels):
        if label == 'yes':
            toxic_words = get_toxic_tokens(text, threshold=0.1)
        else:
            toxic_words = []
        toxic_words_list.append(toxic_words)

    # Free memory after each batch
    del encoding, outputs
    torch.cuda.empty_cache()

    # Periodic garbage collection
    if i % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()

# ============================================
# Save results
# ============================================
df['generation'] = generations
df['label'] = labels
df['toxic_words'] = [str(x) for x in toxic_words_list]

output_file = f"generations/wo-intervention/{MODEL_NAME.split('/')[-1]}_RTP_generations_roberta_toxic_words_extraction.csv"
df[['prompt', 'generation', 'label', 'toxic_words']].to_csv(output_file, index=False)

print(f"\nResults saved to {output_file}")
print(f"Total prompts processed: {len(df)}")
print(f"Toxic generations: {labels.count('yes')}")
