import torch
import numpy as np
import pandas as pd
import tqdm
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

warnings.filterwarnings("ignore")
df = pd.read_csv('datasets/Real_toxic_prompts_toxicit>0.5_5000Sample_rs42.csv')

model_name= 'meta-llama/LLaMA-2-7b-hf'

token = '____YOUR_HF_TOKEN____'

model = AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token


# Load model and tokenizer
cls_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls.eval()

generations = []
labels =[]

for text in tqdm.tqdm(df['prompt']):
  inputs = tokenizer([text] * 25, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
  
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

df.to_json(f"generations/wo-intervention/{model_name.split('/')[-1] if '/' in model_name else model_name}_RTP_generations.json")