# 🚀 KeyGraph — Parallel, TextRank-Inspired Keyword & Phrase Extraction

**PhraseRank** is a blazing-fast, graph-based keyword extraction tool that finds **high-impact phrases** from any text using a **TextRank-inspired algorithm**.

It combines:
- 🧹 **Smart cleaning & tokenization**
- 🏷 **POS tagging & lemmatization**
- ⚡ **Parallel co-occurrence graph building**
- 📈 **PageRank-powered scoring**
- 📝 **Multi-word phrase assembly**

Perfect for **NLP pipelines**, **text summarization**, **topic modeling**, or any application that needs *meaningful keyword phrases* in seconds.

---

## ✨ Features

- **Parallelized Graph Build** — Uses `ThreadPoolExecutor` to scale across CPU cores.
- **POS-Aware Filtering** — Focus on nouns/adjectives (configurable) for relevance.
- **Custom Stopwords** — Supports a `long_stopwords.txt` file for domain tuning.
- **Phrase-Level Output** — Returns contiguous multi-word phrases, not just single tokens.
- **Weighted Edges** — Co-occurrence edges weighted by inverse word distance.
- **Scalable** — Sparse matrices (`scipy.sparse`) keep memory usage low.

---

## 📦 Installation

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scipy nltk
````

**Download NLTK data (first run only)**:

```python
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")  # or "averaged_perceptron_tagger_eng" on newer NLTK
nltk.download("wordnet")
```

---

## 📄 Optional: Stopwords File

Create a file named **`long_stopwords.txt`** in your repo root to add domain-specific stopwords
(e.g., “figure”, “patient”, “dataset”). One word per line.

```
the
of
to
figure
dataset
```

If the file is missing, PhraseRank uses only default punctuation & POS filtering.

---

## ⚡ Usage

```python
from keyword_extraction import find_keywords

text = """
Large language models enable rapid prototyping of information retrieval systems.
We evaluate context windows, reranking, and domain adaptation for medical QA.
"""

results = find_keywords(text, num_keywords=8)

for phrase, score, count in results:
    print(f"{phrase:35s}  score={score:.4f}  count={count}")
```

**Example Output:**

```
language model                    score=2.1832  count=1
information retrieval system      score=1.9479  count=1
domain adaptation                 score=1.2333  count=1
medical qa                        score=0.9937  count=1
```

---

## 🔍 How It Works

1. **Clean & Tokenize** — Removes unwanted characters, tokenizes with NLTK.
2. **POS-Tag & Lemmatize** — Tags words and normalizes forms (e.g., “models” → “model”).
3. **POS-Based Filtering** — Keeps only relevant grammatical categories (configurable).
4. **Build Co-occurrence Graph** — Windowed, inverse-distance weighted.
5. **Parallel Processing** — Chunks text for multi-threaded graph building.
6. **PageRank** — Scores each token by graph centrality.
7. **Assemble Phrases** — Groups contiguous non-stopwords, sums token scores.
8. **Rank & Return** — Outputs top phrases with `(phrase, score, frequency)`.

---

## ⚙️ Configuration

| Parameter      | Description                      | Default                 |
| -------------- | -------------------------------- | ----------------------- |
| `num_keywords` | Max number of phrases to return  | `5`                     |
| `window_size`  | Context window for co-occurrence | `3`                     |
| `num_threads`  | Threads for parallel graph build | `8`                     |
| `keep_tags`    | POS tags to retain               | nouns/adjectives/VBG/FW |

Modify these inside `find_keywords` for customization.

---

## 📂 Project Structure

```
.
├── keyword_extraction.py       # Main keyword extraction logic
├── utils/
│   └── cleaning_utils.py       # Your text cleaning helper
├── long_stopwords.txt          # Optional extra stopwords
├── README.md                   # You are here!
└── requirements.txt
```

---

## 🚀 Why KeyGraph?

* **Fast** — Parallelized co-occurrence computation.
* **Accurate** — POS-aware filtering for better relevance.
* **Flexible** — Easily customize tags, stopwords, and ranking.
* **Lightweight** — No massive ML model dependencies.

---

## 📜 License

MIT License — Free to use, modify, and distribute.

---

💡 *KeyGraph makes your text speak — by finding the words that matter most.*

