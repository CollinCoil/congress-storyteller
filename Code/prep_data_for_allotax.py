"""
Unigram counts for storytellers and non-storytellers, used to build the allotaxonograph (Figure 3).

Input:   Data/testimonies.xlsx
Output:  Data/Storyteller_unigrams.json, Data/Non-Storyteller_unigrams.json
         (in the JSON format read by the Allotaxonometer)
"""
import pandas as pd
import json
import os
from collections import Counter
import string

def process_text_to_tokens(text_series):
    """
    Lowercases, strips punctuation, and splits text into unigram tokens.
    """
    full_text = ' '.join(text_series.dropna().astype(str))
    full_text = full_text.lower()
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    full_text = full_text.translate(translator)
    tokens = full_text.split()
    tokens = [token for token in tokens if token]

    return tokens

def calculate_token_stats(tokens):
    """
    Calculates frequency and probability stats for tokens.
    """
    if not tokens:
        return pd.DataFrame(columns=['types', 'counts', 'totalunique', 'probs'])

    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    total_unique_tokens = len(token_counts)

    stats_data = [
        {
            'types': token,
            'counts': count,
            'totalunique': total_unique_tokens,
            'probs': count / total_tokens
        }
        for token, count in token_counts.items()
    ]

    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values(by='counts', ascending=False).reset_index(drop=True)
    return stats_df

def save_df_to_json(df, file_path):
    """
    Saves a DataFrame to a JSON file in the format required by allotax.
    """
    if not df.empty:
        # Convert DataFrame to a list of dictionaries (records)
        data = df.to_dict('records')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=True, indent=4)
        print(f"Stats saved to '{file_path}'")
    else:
        print(f"DataFrame is empty, no file saved to '{file_path}'")


# --- Main Script ---
file_path = "Data/testimonies.xlsx"
df = pd.read_excel(file_path)

# Ensure the Data directory exists
if not os.path.exists("Data"):
    os.makedirs("Data")

if "Statement" not in df.columns or "Narrative" not in df.columns:
    print(f"Error: The file '{file_path}' must contain 'Statement' and 'Narrative' columns.")
else:
    # Separate the statements into narrative and non-narrative
    non_narrative_statements = df[df["Narrative"] == 0]["Statement"]
    narrative_statements = df[df["Narrative"] == 1]["Statement"]

    print(f"Found {len(non_narrative_statements)} non-narrative statements.")
    print(f"Found {len(narrative_statements)} narrative statements.")

    # --- Case 1: Unigrams (not lemmatized) ---
    print("\nProcessing unigram tokens (not lemmatized)...")
    non_narrative_unigrams = process_text_to_tokens(non_narrative_statements)
    narrative_unigrams = process_text_to_tokens(narrative_statements)

    non_narrative_unigram_stats = calculate_token_stats(non_narrative_unigrams)
    narrative_unigram_stats = calculate_token_stats(narrative_unigrams)

    non_narrative_unigram_json_path = "Data/Non-Storyteller_unigrams.json"
    narrative_unigram_json_path = "Data/Storyteller_unigrams.json"

    save_df_to_json(non_narrative_unigram_stats, non_narrative_unigram_json_path)
    save_df_to_json(narrative_unigram_stats, narrative_unigram_json_path)
