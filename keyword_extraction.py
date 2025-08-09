import string
import time
from typing import List, Dict
import re

import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import lil_matrix
from concurrent.futures import ThreadPoolExecutor

from utils.cleaning_utils import clean_text  # <- your own light cleaner (lowercasing, unicode fixups, etc.)

# ---------------------------
# Helper: build one co-occurrence block for a token chunk
# ---------------------------
def build_partial_matrix(
    processed_text: List[str],
    word_index: Dict[str, int],
    window_size: int
) -> lil_matrix:
    """
    Build a sparse co-occurrence matrix for a list of tokens using a sliding window.

    Args:
        processed_text: preprocessed tokens (stopwords/punct filtered, lemmatized)
        word_index: mapping token -> integer index
        window_size: number of tokens to consider in the local context

    Returns:
        lil_matrix: (vocab_len x vocab_len) weighted co-occurrence counts
    """
    # Number of unique tokens in the shared vocabulary
    vocab_len = len(word_index)

    # LIL is efficient for incremental (row-wise) construction
    partial_matrix = lil_matrix((vocab_len, vocab_len), dtype=np.float32)

    # Slide a fixed-size window across this chunk
    for start_i in range(len(processed_text) - window_size + 1):
        window_words = processed_text[start_i : start_i + window_size]

        # For every pair (i, j) inside this window…
        for j in range(window_size):
            idx_i = word_index[window_words[j]]
            for k in range(j + 1, window_size):
                idx_j = word_index[window_words[k]]

                # Skip self-loops
                if idx_i == idx_j:
                    continue

                # Inverse-distance weight (closer words reinforce more)
                dist = abs(j - k) or 1
                w = 1.0 / dist

                # Symmetric update (undirected graph)
                partial_matrix[idx_i, idx_j] += w
                partial_matrix[idx_j, idx_i] += w

    return partial_matrix


# ---------------------------
# Main: extract keywords with a TextRank-inspired pipeline
# ---------------------------
def find_keywords(
    text: str,
    num_keywords: int = 5
) -> List[str]:
    """
    Extract top keyword *phrases* using:
      - cleaning + tokenization
      - POS-tagging + lemmatization
      - POS-based filtering + custom stopwords
      - parallel co-occurrence graph build
      - PageRank on transition matrix
      - phrase assembly + scoring

    Args:
        text: raw document text
        num_keywords: maximum number of phrases to return

    Returns:
        List[Tuple[str, float, int]] where:
            - str: keyword phrase
            - float: phrase score
            - int: exact match count in cleaned text
    """
    start_time = time.time()

    # --- 1) Clean + tokenize ---
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    if not tokens:
        return []

    # --- 2) POS-tag + lemmatize ---
    # First tag the original tokens
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize; adjectives use "a" POS to reduce forms like 'larger' -> 'large'
    lemmatizer = WordNetLemmatizer()
    adjective_tags = {"JJ", "JJR", "JJS"}
    lemmatized_text = [
        lemmatizer.lemmatize(word, "a") if tag in adjective_tags else lemmatizer.lemmatize(word)
        for word, tag in pos_tags
    ]

    # Re-tag the lemmatized tokens (helps after normalization)
    pos_tags_2 = nltk.pos_tag(lemmatized_text)

    # --- 3) POS filtering + stopwords/punct removal ---
    # Keep mostly nouns/adjectives, allow VBG (gerunds) and FW (foreign words)
    keep_tags = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VBG", "FW"}
    removed_words = [w for (w, t) in pos_tags_2 if t not in keep_tags]

    # Punctuation list and optional long stoplist from file
    punctuation = list(string.punctuation)
    additional_stopwords = []
    try:
        with open("long_stopwords.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    additional_stopwords.append(line)
    except FileNotFoundError:
        # Optional file. If absent we proceed with default stoplist.
        pass

    stopwords_plus = set(removed_words + punctuation + additional_stopwords)

    # Final token list used to build the graph
    processed_text = [w for w in lemmatized_text if w not in stopwords_plus]
    if not processed_text:
        return []

    # --- 4) Build vocabulary + word index ---
    vocabulary = list(set(processed_text))
    vocab_len = len(vocabulary)
    word_index = {word: idx for idx, word in enumerate(vocabulary)}

    # Context window (3 is a good small default; tune for your corpus)
    window_size = 3

    # --- Parallel chunking ---
    # Use multiple threads to build separate co-occurrence blocks, then sum them.
    num_threads = 8
    chunk_size = max(1, len(processed_text) // num_threads)
    chunks = [processed_text[i : i + chunk_size] for i in range(0, len(processed_text), chunk_size)]

    # --- 5) Build co-occurrence matrices in parallel ---
    matrix_start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        partial_matrices = list(
            executor.map(
                build_partial_matrix,
                chunks,
                [word_index] * len(chunks),   # match length of chunks
                [window_size] * len(chunks)
            )
        )
    matrix_end_time = time.time()

    # Combine into a single sparse matrix (weighted edge matrix)
    weighted_edge = sum(partial_matrices).tocsr()

    # --- 6) Normalize columns to get a column-stochastic transition matrix T ---
    # inout[j] = total outgoing weight from node j
    inout = np.array(weighted_edge.sum(axis=1)).flatten()
    inout[inout == 0] = 1e-9  # avoid divide-by-zero for isolated nodes

    # Normalize *columns* efficiently by working in CSC form
    T_csc = weighted_edge.tocsc()
    for j in range(vocab_len):
        start_ptr, end_ptr = T_csc.indptr[j], T_csc.indptr[j + 1]
        if inout[j] != 0:
            T_csc.data[start_ptr:end_ptr] /= inout[j]
    T = T_csc.tocsr()

    # --- 7) PageRank iteration: score = (1-d) + d * T @ score ---
    damping_factor = 0.85
    max_iterations = 50
    threshold = 1e-4

    score = np.ones(vocab_len, dtype=np.float32)
    for _ in range(max_iterations):
        prev_score = score.copy()
        score = (1 - damping_factor) + damping_factor * (T.dot(prev_score))
        if np.sum(np.abs(prev_score - score)) <= threshold:
            break

    # --- 8) Assemble phrases by grouping contiguous non-stopwords in the original lemmatized stream ---
    phrases: List[List[str]] = []
    current_phrase: List[str] = []
    for w in lemmatized_text:
        if w in stopwords_plus:
            if current_phrase:
                phrases.append(current_phrase)
            current_phrase = []
        else:
            current_phrase.append(w)
    if current_phrase:
        phrases.append(current_phrase)

    # Deduplicate phrases (keep first occurrence)
    unique_phrases: List[List[str]] = []
    seen = set()
    for ph in phrases:
        p_str = " ".join(ph)
        if p_str not in seen and len(ph) > 0:
            seen.add(p_str)
            unique_phrases.append(ph)

    # --- 9) Score phrases as sum of token scores ---
    phrase_scores: List[float] = []
    keywords: List[str] = []
    for ph in unique_phrases:
        p_score = sum(score[word_index[w]] for w in ph if w in word_index)
        phrase_scores.append(p_score)
        keywords.append(" ".join(ph))

    # --- 10) Sort and pick top K; also count exact matches in cleaned text for reporting ---
    sorted_index = np.flip(np.argsort(phrase_scores), 0)

    keyword_result = []
    for i in range(min(num_keywords, len(sorted_index))):
        idx = sorted_index[i]
        phrase = keywords[idx]
        pscore = phrase_scores[idx]
        # Count full-word matches; allows suffix characters with \w*
        count = len(re.findall(r"\b" + re.escape(phrase) + r"\w*", cleaned_text))
        keyword_result.append((phrase, pscore, count))

    return keyword_result
