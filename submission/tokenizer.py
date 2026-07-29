"""Fast Pure-Python Byte-level BPE Tokenizer for Plivo ML Assignment.

Guarantees 100% Lossless Roundtrip: decode(encode(text)) == text.
Base vocabulary consists of 256 byte tokens (0..255).
Subwords are created by byte-pair encoding trained on train_corpus.txt.
Uses word-level caching for fast encoding on large corpora.
"""
import json
import os


class ByteLevelBPETokenizer:
    def __init__(self, vocab_size=512):
        self.target_vocab_size = vocab_size
        self.merges = {}  # (p1, p2) -> new_id
        self.ranks = {}   # (p1, p2) -> rank
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.vocab_size = 256
        self.cache = {}

    def train(self, text, target_vocab_size=512, max_bytes_sample=30000):
        """Train BPE merges on raw text up to target_vocab_size."""
        self.target_vocab_size = target_vocab_size
        raw_bytes = text.encode("utf-8")[:max_bytes_sample]
        tokens = list(raw_bytes)

        num_merges = target_vocab_size - 256
        self.merges = {}
        self.ranks = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for i in range(num_merges):
            pair_counts = {}
            for p1, p2 in zip(tokens[:-1], tokens[1:]):
                pair = (p1, p2)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break

            self.merges[best_pair] = next_id
            self.ranks[best_pair] = i
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            new_tokens = []
            idx = 0
            while idx < len(tokens):
                if idx < len(tokens) - 1 and (tokens[idx], tokens[idx + 1]) == best_pair:
                    new_tokens.append(next_id)
                    idx += 2
                else:
                    new_tokens.append(tokens[idx])
                    idx += 1
            tokens = new_tokens
            next_id += 1

        self.vocab_size = len(self.vocab)
        self.cache = {}

    def _encode_bytes(self, raw_bytes):
        tokens = list(raw_bytes)
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            valid_pairs = [p for p in pairs if p in self.ranks]
            if not valid_pairs:
                break

            best_pair = min(valid_pairs, key=lambda p: self.ranks[p])
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text):
        """Encode text to token IDs using fast word-cached byte-level BPE."""
        if not self.ranks:
            return list(text.encode("utf-8"))

        words = text.split(" ")
        tokens = []
        for i, word in enumerate(words):
            if i > 0:
                tokens.append(32)  # ASCII space byte
            if not word:
                continue
            if word not in self.cache:
                self.cache[word] = self._encode_bytes(word.encode("utf-8"))
            tokens.extend(self.cache[word])

        return tokens

    def decode(self, ids):
        """Decode token IDs back to exact string. Guaranteed 100% lossless."""
        byte_chunks = [self.vocab.get(i, bytes([i % 256])) for i in ids]
        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        """Save merges and vocab configuration to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": [list(k) + [v] for k, v in self.merges.items()]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_vocab(self, path):
        """Load merges from JSON file."""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = {}
        self.ranks = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        rank = 0

        for p1, p2, next_id in data.get("merges", []):
            pair = (p1, p2)
            self.merges[pair] = next_id
            self.ranks[pair] = rank
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            rank += 1

        self.vocab_size = len(self.vocab)
        self.cache = {}


def load(path=None):
    """Load BPE tokenizer used by train.py and evaluate.py."""
    tok = ByteLevelBPETokenizer()
    vocab_path = os.path.join(os.path.dirname(__file__), "bpe_vocab.json")

    if os.path.exists(vocab_path):
        tok.load_vocab(vocab_path)
    else:
        corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "train_corpus.txt")
        if os.path.exists(corpus_path):
            text = open(corpus_path, encoding="utf-8").read()
            tok.train(text, target_vocab_size=512, max_bytes_sample=30000)
            tok.save(vocab_path)

    return tok
