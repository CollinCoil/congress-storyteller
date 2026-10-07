"""
Testimony- and hearing-level information content scores.

Each statement is split into sentences with Stanza and embedded with Qwen/Qwen3-Embedding-4B.
Sentences are scored in order: the first sentence scores 1, and each later sentence scores its
minimum angular distance to all preceding sentences. A testimony's score is the sum over its
sentences. A hearing's score is computed the same way over all sentences from its witnesses.

Input:   Data/testimonies.xlsx
Output:  Data/witness_testimony_content.xlsx, Data/hearing_testimony_content.xlsx
Note:    A GPU with roughly 16 GB of memory is recommended.
"""
import pandas as pd
import numpy as np
import stanza
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict
import torch

# Initialize Stanza pipeline
stanza.download("en")
nlp = stanza.Pipeline("en", processors="tokenize", tokenize_pretokenized=False)

def information_content(embeddings):
    """Sum of each sentence's minimum angular distance to the preceding sentences (first sentence = 1)."""
    if len(embeddings) == 0:
        return 0.0
    
    # Normalize embeddings to unit vectors
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    score = 1.0
    for i in range(1, len(embeddings_norm)):
        # Calculate cosine similarities for the i-th embedding against all previous ones
        cosine_similarities = np.dot(embeddings_norm[i], embeddings_norm[:i].T)
        
        # Clamp values to the valid range for arccos
        cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
        
        # Calculate angular differences and find the minimum
        angular_diffs = np.arccos(cosine_similarities) / np.pi
        score += np.min(angular_diffs)
        
    return score

# Load data
data = pd.read_excel("Data/testimonies.xlsx")

# Preprocess: split into sentences and store
all_sentences = []
witness_sentence_indices = []

for index, row in data.iterrows():
    doc = nlp(row['Statement'])
    sentences = [s.text for s in doc.sentences]
    all_sentences.extend(sentences)
    witness_sentence_indices.append((index, len(sentences)))

# Model setup: This is the new, simplified part
model_name = "Qwen/Qwen3-Embedding-4B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")
model = SentenceTransformer(model_name, device=device)

# Generate sentence embeddings with a single function call
sentence_embeddings = model.encode(all_sentences, show_progress_bar=True, convert_to_tensor=False, batch_size=128)

# Track per-witness and per-hearing information content
witness_scores = []
hearing_embeddings = defaultdict(list)
hearing_data = defaultdict(lambda: {
    'Num_Witnesses': 0,
    'Num_Women': 0,
    'Occupations': Counter(),
    'Num_Narratives': 0
})
# Hearing type is constant within a hearing, so it is recorded once per hearing
hearing_types = {}

offset = 0
for index, num_sentences in witness_sentence_indices:
    row = data.iloc[index]
    witness_points = sentence_embeddings[offset:offset+num_sentences]
    score = information_content(witness_points)
    witness_scores.append(score)

    key = (row['Hearing'], row['Committee'], row['Date'], row['Branch'], row['Subcommittee'])
    hearing_embeddings[key].extend(witness_points)
    hearing_types[key] = row['Hearing_Type']

    stats = hearing_data[key]
    stats['Num_Witnesses'] += 1
    if row['Gender'] == 1:
        stats['Num_Women'] += 1
    stats['Occupations'][row['Occupation']] += 1
    if row['Narrative']:
        stats['Num_Narratives'] += 1

    offset += num_sentences

# Compute hearing-level information content
hearing_content = []
for key, points in hearing_embeddings.items():
    score = information_content(points)
    stats = hearing_data[key]
    stats['Info_Content'] = score
    stats['Percent_Women'] = stats['Num_Women'] / stats['Num_Witnesses']
    stats['Herfindahl_Index'] = sum((count / stats['Num_Witnesses']) ** 2 for count in stats['Occupations'].values())
    stats['Percent_Narratives'] = stats['Num_Narratives'] / stats['Num_Witnesses']
    hearing_content.append({**stats, 'Hearing': key[0], 'Hearing_Type': hearing_types[key], 'Committee': key[1], 'Date': key[2], 'Branch': key[3], 'Subcommittee': key[4]})

# Save outputs
witness_df = data.copy()
witness_df['Info_Content'] = witness_scores
witness_df.to_excel("Data/witness_testimony_content.xlsx", index=False)

hearing_df = pd.DataFrame(hearing_content)
hearing_df.to_excel("Data/hearing_testimony_content.xlsx", index=False)