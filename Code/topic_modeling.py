"""
NMF topic modeling of storyteller and non-storyteller testimony (Figures 4-7).

Selects the number of topics by C_v coherence, fits the final NMF model, and compares the
topic distributions of the two groups with Jensen-Shannon divergence, a permutation test,
and bootstrap confidence intervals.

Input:   Data/testimonies.xlsx
Output:  Results/optimal_number_topics.png (Figure 4)
         Results/permutation_test_jsd.png (Figure 5)
         Results/topic_wise_jsd.png (Figure 6)
         Results/topic_probabilities.png (Figure 7)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from scipy.stats import entropy as scipy_entropy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import random

os.makedirs("Results", exist_ok=True)

# ----------------------------
# Step 1: Preprocessing utils
# ----------------------------

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(documents):
    processed_docs = []
    processed_texts = []
    for doc in documents:
        tokens = nltk.word_tokenize(doc.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_docs.append(tokens)
        processed_texts.append(' '.join(tokens))
    return processed_docs, processed_texts

# ----------------------------
# Step 2: NMF Model Training
# ----------------------------
def train_nmf(processed_texts, num_topics, random_state=2025):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Train NMF model
    nmf_model = NMF(n_components=num_topics, random_state=random_state, max_iter=1000)
    document_topics = nmf_model.fit_transform(tfidf_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    return nmf_model, tfidf_matrix, vectorizer, document_topics, feature_names

# ----------------------------
# Step 3: Determine optimal number of topics
# ----------------------------
def determine_optimal_topics(processed_docs, range_start=3, range_end=15, random_state=2025, n_top_words=10):
    """
    Determine optimal number of topics for NMF using coherence score (CV).
    processed_docs: list of tokenized documents (list of lists of strings)
    """
    topic_results = []
    
    # Prepare corpus for coherence calculation
    dictionary = Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(text) for text in processed_docs]
    
    # Join tokens back into strings for TfidfVectorizer
    processed_texts_for_tfidf = [" ".join(doc) for doc in processed_docs]
    
    for num_topics in range(range_start, range_end + 1):
        # Vectorize text
        vectorizer = TfidfVectorizer(min_df=3, max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(processed_texts_for_tfidf)
        
        # Fit NMF
        nmf_model = NMF(n_components=num_topics, random_state=random_state, max_iter=1000)
        nmf_model.fit(tfidf_matrix)
        
        # Extract top words per topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(top_features)
        
        # Compute coherence score
        cm = CoherenceModel(topics=topics, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        coherence_score = cm.get_coherence()
        
        topic_results.append((num_topics, coherence_score))
        print(f"Num Topics = {num_topics}, Coherence Score = {coherence_score:.4f}")
    
    # Select best number of topics
    best_num_topics = max(topic_results, key=lambda x: x[1])[0]
    return best_num_topics, topic_results

# ----------------------------
# Step 4: Print Topics
# ----------------------------
def print_topics(nmf_model, feature_names, n_top_words=5):
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words-1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        topic_str = " + ".join([f"{weights[i]:.3f}*{feature}" for i, feature in enumerate(top_features)])
        topics.append((topic_idx, topic_str))
        
    return topics

# ----------------------------
# Step 5: Get Topic Distributions
# ----------------------------
def get_topic_distributions(document_topics):
    # Normalize topic distributions to sum to 1 for each document
    topic_distributions = []
    for doc_topic in document_topics:
        distribution = doc_topic / doc_topic.sum() if doc_topic.sum() > 0 else doc_topic
        topic_distributions.append(distribution)
    return np.array(topic_distributions)

# ----------------------------
# Step 6: Aggregate Topic Distributions
# ----------------------------
def aggregate_topic_distributions(topic_distributions, labels, target_label):
    subset = topic_distributions[np.array(labels) == target_label]
    distribution = np.mean(subset, axis=0)
    return distribution / distribution.sum() if distribution.sum() > 0 else distribution

# ----------------------------
# Step 7: JS Divergence & Decomposition
# ----------------------------
def compute_js_divergence(p, q):
    m = 0.5 * (p + q)
    js = 0.5 * (scipy_entropy(p, m) + scipy_entropy(q, m))
    contributions = 0.5 * (p * (np.log(p / m + 1e-12)) + q * (np.log(q / m + 1e-12)))
    return js, contributions

# ----------------------------
# Step 8: Bootstrap Confidence Intervals
# ----------------------------
def bootstrap_contributions(topic_distributions, labels, target_labels, n_bootstrap=5000):
    bootstrapped_contributions = []
    for _ in tqdm(range(n_bootstrap)):
        indices = np.random.choice(len(labels), size=len(labels), replace=True)
        sample_distributions = topic_distributions[indices]
        sample_labels = np.array(labels)[indices]
        p = aggregate_topic_distributions(sample_distributions, sample_labels, target_labels[0])
        q = aggregate_topic_distributions(sample_distributions, sample_labels, target_labels[1])
        _, contributions = compute_js_divergence(p, q)
        bootstrapped_contributions.append(contributions)
    bootstrapped_contributions = np.array(bootstrapped_contributions)
    lower = np.percentile(bootstrapped_contributions, 2.5, axis=0)
    upper = np.percentile(bootstrapped_contributions, 97.5, axis=0)
    mean = np.mean(bootstrapped_contributions, axis=0)
    return mean, lower, upper

# ----------------------------
# Step 9: Permutation Test
# ----------------------------
def permutation_test(topic_distributions, labels, target_labels, n_permutations=5000):
    observed_p = aggregate_topic_distributions(topic_distributions, labels, target_labels[0])
    observed_q = aggregate_topic_distributions(topic_distributions, labels, target_labels[1])
    observed_js, _ = compute_js_divergence(observed_p, observed_q)

    permuted_js = []
    for _ in tqdm(range(n_permutations)):
        permuted_labels = shuffle(labels, random_state=random.randint(0, 100000))
        p = aggregate_topic_distributions(topic_distributions, permuted_labels, target_labels[0])
        q = aggregate_topic_distributions(topic_distributions, permuted_labels, target_labels[1])
        js, _ = compute_js_divergence(p, q)
        permuted_js.append(js)

    p_value = np.mean(np.array(permuted_js) >= observed_js)
    return observed_js, permuted_js, p_value

# ----------------------------
# Step 10: Bootstrap Differences
# ----------------------------
def bootstrap_differences(topic_distributions, labels, target_labels, n_bootstrap=5000):
    bootstrapped_differences = []
    for _ in tqdm(range(n_bootstrap)):
        indices = np.random.choice(len(labels), size=len(labels), replace=True)
        sample_distributions = topic_distributions[indices]
        sample_labels = np.array(labels)[indices]
        p = aggregate_topic_distributions(sample_distributions, sample_labels, target_labels[0])
        q = aggregate_topic_distributions(sample_distributions, sample_labels, target_labels[1])
        differences = p - q
        bootstrapped_differences.append(differences)
    bootstrapped_differences = np.array(bootstrapped_differences)
    lower = np.percentile(bootstrapped_differences, 2.5, axis=0)
    upper = np.percentile(bootstrapped_differences, 97.5, axis=0)
    mean = np.mean(bootstrapped_differences, axis=0)
    return mean, lower, upper

# ----------------------------
# Step 11: Plot Topic Selection Results
# ----------------------------
def plot_topic_selection_results(topic_results):
    num_topics = [result[0] for result in topic_results]
    scores = [result[1] for result in topic_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Number of Topics')
    ax1.set_ylabel('Coherence Score', color=color)
    ax1.plot(num_topics, scores, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    best_topic_idx = scores.index(max(scores))
    best_num_topics = num_topics[best_topic_idx]
    plt.axvline(x=best_num_topics, linestyle='--', color='gray', alpha=0.7)
    plt.title('Coherence Score vs Number of Topics')
    
    plt.tight_layout()
    plt.savefig("Results/optimal_number_topics.png")
    
    return best_num_topics

# Create a helper function to get topic labels with top words
def get_topic_labels_with_top_words(nmf_model, feature_names, n_top_words=3):
    topic_labels = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words-1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_label = f"Topic {topic_idx}: {('_').join(top_features)}"
        topic_labels.append(topic_label)
    return topic_labels

# ----------------------------
# Step 12: Main Function
# ----------------------------
def main():
    # Load data
    data = pd.read_excel("Data/testimonies.xlsx")
    documents = data["Statement"].dropna().tolist()
    labels = data["Narrative"].dropna().tolist()

    # Preprocess documents
    processed_docs, processed_texts = preprocess(documents)

    # Determine number of topics using coherence score
    best_num_topics, topic_results = determine_optimal_topics(processed_docs, range_start=5, range_end=30)
    print(f"Optimal number of topics determined: {best_num_topics}")
    
    # Plot topic selection results
    plot_topic_selection_results(topic_results)

    # Train final NMF model
    # Use processed_texts for the final model
    nmf_model, tfidf_matrix, vectorizer, document_topics, feature_names = train_nmf(processed_texts, best_num_topics)
    
    # Normalize document-topic distributions
    topic_distributions = get_topic_distributions(document_topics)

    # Print topics
    topics = print_topics(nmf_model, feature_names, n_top_words=20)
    for topic_num, topic in topics:
        print(f"Topic #{topic_num}: {topic}")

    target_labels = list(dict.fromkeys(labels))

    # JS Divergence and decomposition
    p = aggregate_topic_distributions(topic_distributions, labels, target_labels[0])
    q = aggregate_topic_distributions(topic_distributions, labels, target_labels[1])
    js_div, contributions = compute_js_divergence(p, q)

    print(f"Jensen-Shannon Divergence: {js_div}")

    # Bootstrap confidence intervals
    mean_contrib, lower_contrib, upper_contrib = bootstrap_contributions(topic_distributions, labels, target_labels)

    # Visualize contributions
    plt.figure(figsize=(12, 6))
    topics = get_topic_labels_with_top_words(nmf_model, feature_names, n_top_words=3)
    plt.bar(topics, mean_contrib, yerr=[mean_contrib - lower_contrib, upper_contrib - mean_contrib], capsize=5)
    plt.xticks(rotation=45, ha='right')
    plt.title("Topic-wise Contributions to JS Divergence (with 95% CI)")
    plt.tight_layout()
    plt.savefig("Results/topic_wise_jsd.png")


    # Permutation test
    observed_js, permuted_js, p_value = permutation_test(topic_distributions, labels, target_labels)

    # Visualize permutation test
    plt.figure(figsize=(8, 6))
    sns.histplot(permuted_js, bins=30, kde=True)
    plt.axvline(observed_js, color='red', linestyle='--', label=f'Observed JS: {observed_js:.4f}')
    plt.title(f'Permutation Test (p-value = {p_value:.4f})')
    plt.legend()
    plt.xlabel('JS Divergence')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("Results/permutation_test_jsd.png")


    print(f"Permutation test p-value: {p_value}")

    # Bootstrap differences
    mean_diff, lower_diff, upper_diff = bootstrap_differences(topic_distributions, labels, target_labels)

    # Visualize differences
    plt.figure(figsize=(12, 6))
    plt.bar(topics, mean_diff, yerr=[mean_diff - lower_diff, upper_diff - mean_diff], capsize=5)
    plt.xticks(rotation=45, ha='right')
    plt.title("Differences in Topic Probabilities (with 95% CI)")
    plt.tight_layout()
    plt.savefig("Results/topic_probabilities.png")


if __name__ == "__main__":
    main()