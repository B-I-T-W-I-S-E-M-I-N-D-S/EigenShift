import torch
import pandas as pd
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import Saliency
from torch.nn.functional import softmax
import warnings
warnings.filterwarnings("ignore")

model_name= 'meta-llama/LLaMA-2-7b-hf'

token = '____YOUR_HF_TOKEN____'

model = AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token

# Load Roberta Toxicity Classifier
cls_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls_model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')
cls_model.eval().to("cuda:1")

df = pd.read_csv('/home/abdullahm/NPS25/step-1-Initial_experiments/llama-2-7b_RTP_5000_generations.csv')

generations = []
labels = []
toxic_words_list = []
batch_size = 1
after_tokenizer = []
saliency = Saliency(cls_model)

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

def get_toxic_tokens(text):
    encoding = cls_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=30).to("cuda:1")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    input_embed = cls_model.roberta.embeddings(input_ids)
    input_embed = input_embed.detach()
    input_embed.requires_grad = True

    wrapper = RobertaWrapper(cls_model, attention_mask)

    saliency = Saliency(wrapper)
    grads = saliency.attribute(inputs=input_embed, target=1, abs=True)

    grads_sum = grads.sum(dim=-1).squeeze(0)
    tokens = cls_tokenizer.convert_ids_to_tokens(input_ids[0])
    topk = torch.topk(grads_sum, k=1)

    top_tokens = []
    for i in topk.indices:
        if grads_sum[i] > 0.1:
            token = tokens[i]
            token_str = cls_tokenizer.convert_tokens_to_string([token]).strip()
            if token_str and token_str in text:
                top_tokens.append(token_str)

    return top_tokens



for i in tqdm.tqdm(range(0, len(df), batch_size)):
    batch_prompts = df['prompt'][i:i+batch_size].tolist()
    inputs = tokenizer(batch_prompts, return_tensors="pt", truncation=True).to("cuda:0")
    after_tokenizer.append(inputs)
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
            top_p=None
        )
    output_texts = [
        tokenizer.decode(output, skip_special_tokens=True).split(prompt)[-1].strip()
        for output, prompt in zip(output_tokens, batch_prompts)
    ]

    generations.extend(output_texts)
    encoding = cls_tokenizer(output_texts, return_tensors="pt", padding=True, truncation=True, max_length=30).to("cuda:1")
    with torch.no_grad():
        outputs = cls_model(**encoding)
        preds = torch.argmax(outputs.logits, dim=1).tolist()

    batch_labels = ['yes' if p == 1 else 'no' for p in preds]
    labels.extend(batch_labels)

    for text, label in zip(output_texts, batch_labels):
        if label == 'yes':
            toxic_words = get_toxic_tokens(text)
        else:
            toxic_words = []
        toxic_words_list.append(toxic_words)
    print(toxic_words)
df['generation'] = generations
df['label'] = labels
df['toxic_words'] = [f'{x}' for x in toxic_words_list]

df[['prompt', 'generation', 'label', 'toxic_words']].to_csv(f"generations/wo-intervention{model_name.split('/')[-1]}_RTP_generations_roberta_toxic_words_extraction.csv", index=False)