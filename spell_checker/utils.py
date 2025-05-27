# spell_checker/utils.py
import re

def normalize_token(token: str) -> str:
    token = token.lower()
    token = token.replace("’", "'").replace("‘", "'").replace("ʻ", "'").replace("ʼ", "'")
    token = re.sub(r"^[^a-zA-Z0-9ʻʼ']+", "", token)
    token = re.sub(r"[^a-zA-Z0-9ʻʼ']+$", "", token)
    token = re.sub(r"[^\w'ʻʼ-]", "", token)
    return token

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)
    prev_row = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr_row = [i + 1]
        for j, c2 in enumerate(b):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            replace = prev_row[j] + (c1 != c2)
            curr_row.append(min(insert, delete, replace))
        prev_row = curr_row
    return prev_row[-1]
