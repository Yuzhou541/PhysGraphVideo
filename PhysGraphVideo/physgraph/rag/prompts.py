import re
def to_tokens(texts, vocab=None, max_len=32):
    V = 4096 if vocab is None else len(vocab)
    ids = []
    for t in texts:
        toks = re.findall(r"[a-zA-Z]+", t.lower())
        arr = [(hash(tok) % V) for tok in toks][:max_len]
        arr = arr + [0]*(max_len-len(arr))
        ids.append(arr)
    return ids
