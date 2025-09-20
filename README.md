# EigenShift: Eigen-based Intervention for Toxicity Reduction in LLMs

## Step 0: Setup

Create and activate a virtual environment using Python 3.8.10:

```bash
python3.8 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 1: Generate Outputs from the Model

To generate outputs from your model, navigate to the following script:

```bash
cd EigenShift
python get_generations.py
```

Before running, make sure to update the following in `get_generations.py`:

- `model_name`
- Hugging Face `token`
- Device (e.g., `"cuda"` or `"cpu"`)

This script will save generations to:

```
EigenShift/generations/wo-intervention
```

To simplify the process, we've already run this script and saved the output at:

```
EigenShift/generations/wo-intervention/LLaMA-2-7b-hf_RTP_generations.json
```

This file contains 5000 toxic generations from the LLaMA 2 7B model using RealToxicPrompts (RTP).

---

## Step 2: Extract Toxic Words

We use a pre-trained toxicity classifier (`s-nlp/roberta_toxicity_classifier`) to identify toxic words in the generations.

To run the script:

```bash
python toxic_words_extraction.py
```

Make sure to configure the model/token if needed.

We’ve also saved the processed output for convenience:

```
EigenShift/generations/wo-intervention/LLaMA-2-7b-hf_RTP_generations_roberta_toxic_words_extraction.csv
```

---

## Step 3: Intervention via Matrix Reconstruction

This step performs the core methodology:

- Builds toxic/non-toxic hidden state clusters.
- Applies matrix factorization (SVD) on `lm_head`.
- Projects hidden states onto eigenvectors.
- Computes delta scores (toxicity alignment).
- Dampens eigenvectors based on toxicity and reconstructs `lm_head`.

To run this process:

```bash
python reconstruct.py
```

---

## Step 4: Evaluate After Intervention

Replace the `lm_head` with the reconstructed one and evaluate the updated model on RTP dataset:

```bash
python evaluation_after_intervention.py
```

---

## Baselines

For comparison against baseline methods, we used the official implementation from:

[https://github.com/apple/ml-aura](https://github.com/apple/ml-aura)

---

## Citation

If this work is helpful in your research, please cite:

> BibTeX coming soon.
