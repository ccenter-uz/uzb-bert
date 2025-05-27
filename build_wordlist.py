import re
import json
from collections import Counter
from datasets import load_dataset
import pandas as pd

# === CONFIGURATION ===
MIN_COUNT = 3
MIN_LENGTH = 2
CACHE_DIR = "data"
OUTPUT_PATH = "./spell_checker/uz_words.json"
UCHAR_WHITELIST = "a-zA-Z0-9ʻʼ'-"

# === DOMAIN FILTER ===
domain_prefixes = pd.read_csv("distinct_domains.csv")["domain"].tolist()
domain_prefixes = set(domain_prefixes)  # O(1) lookup

def is_allowed_source(source):
    return any(domain in source for domain in domain_prefixes)

# === TOKENIZER ===
_token_cleaner = re.compile(rf"[^\w{UCHAR_WHITELIST}]")
_leading_trailing = re.compile(r"^[^a-zA-Z0-9ʻʼ']+|[^a-zA-Z0-9ʻʼ']+$")

def normalize_token(token):
    token = token.replace("’", "'").replace("‘", "'").replace("ʻ", "'").replace("ʼ", "'")
    token = _leading_trailing.sub("", token)
    token = _token_cleaner.sub("", token)
    return token if len(token) >= 2 and not re.fullmatch(r"[\d\W]+", token) else ""

def tokenize(text):
    return [normalize_token(t) for t in text.split() if normalize_token(t)]

# === MAIN ===
def build_wordlist_from_gov_only():
    print("📥 Loading dataset from Hugging Face...")
    ds = load_dataset("tahrirchi/uz-crawl", cache_dir=CACHE_DIR)
    articles = ds["news"]

    print("⚙️ Filtering and tokenizing...")
    word_counter = Counter()

    for article in articles:
        if is_allowed_source(article["source"]):
            tokens = tokenize(article["text"])
            word_counter.update(tokens)
            print(f"  - {article['source']}: {len(tokens)} tokens")

    print(f"📊 Extracted {len(word_counter)} unique tokens before filtering... (includes)")

    final_words = sorted([
        word for word, count in word_counter.items()
        if count >= MIN_COUNT
        and len(word) >= MIN_LENGTH
        and word[0].isalpha()
        and not re.search(r"\d{3,}", word)
    ])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_words, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved cleaned wordlist with {len(final_words)} words → {OUTPUT_PATH}")

if __name__ == "__main__":
    build_wordlist_from_gov_only()
