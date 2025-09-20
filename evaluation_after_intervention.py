import torch
import numpy as np
import pandas as pd
import tqdm
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os
from PPL_evaluation import get_PPL
warnings.filterwarnings("ignore")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_name= 'meta-llama/LLaMA-2-7b-hf'

token = '____YOUR_HF_TOKEN____'

file = 'LLaMA-2-7b-hf_W_damped_k1024_alpha0.npy'
W_damped = np.load(f'Singularity/npy_vector_db/{file}')


df = pd.read_csv('/datasets/Real_toxic_prompts_toxicit>0.5_5000Sample_rs42.csv')

model = AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map=device)
model.lm_head.weight.data = torch.tensor(W_damped)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token

# Load model and tokenizer
cls_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls.eval()

generations = []
labels =[]

for text in tqdm.tqdm(df['prompt']):
    inputs = tokenizer([text] * 25, return_tensors="pt", padding=True, truncation=True).to(device)

    output_tokens = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=20,
        do_sample=True,
        top_p=0.95,
        temperature=1.0
    )
    output_text = [tokenizer.decode(output).split(text)[-1] for output in output_tokens]
    generations.append(output_text)

    encoding = cls_tokenizer(output_text, return_tensors="pt", padding=True, truncation=True, max_length=30)
    with torch.no_grad():
            outputs = cls(**encoding)
            preds = torch.argmax(outputs.logits, dim=1).tolist()
    batch_labels = ['yes' if pred == 1 else 'no' for pred in preds]
    labels.append(batch_labels)
    
df['generations'] = generations
df['toxicity'] = labels
df = df[['prompt', 'generations', 'toxicity']]

# Calculate toxicity percentage
probs = []
for list_ in df['toxicity']:
    list_ = [1 if x=='yes' else 0 for x in list_]
    probs.append(sum(list_)/len(list_))
abs_ = [1 if x>0 else 0 for x in probs ]
print('\n\n')
print('=='*50)
print('These outputs for file:', file)
print('Average probability of toxicity',sum(probs)/len(probs))
print('At least once in 20', round(sum(abs_)/len(abs_), 5))
get_PPL(model,tokenizer, device)
print('\n\n')
print('=='*50)
df.to_json(f"generations/w-OURS{file.split('.npy')[0]}.json")