"""Byte BPE with deterministic tie-breaking and incremental pair statistics."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import base64
import heapq
import json
import regex

PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def special_pattern(tokens):
    return regex.compile('(' + '|'.join(regex.escape(s) for s in sorted(tokens, key=lambda s: (-len(s), s))) + ')') if tokens else None


def merge_word(word, pair):
    result, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            result.append(word[i] + word[i + 1])
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


class ReversePair:
    """Heap tie-break: lexicographically largest byte pair wins."""
    __slots__ = ('pair',)
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair
    def __eq__(self, other):
        return self.pair == other.pair


def train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    special_tokens = list(dict.fromkeys(special_tokens))
    if vocab_size < 256 + len(special_tokens):
        raise ValueError('Vocabulary must contain all bytes and special tokens')
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    splitter = special_pattern(special_tokens)
    raw_counts = Counter()
    # Documents are split before pretokenization; specials never enter pair counts.
    text = Path(input_path).read_text(encoding='utf-8')
    chunks = splitter.split(text) if splitter else [text]
    special_set = set(special_tokens)
    for chunk in chunks:
        if chunk in special_set:
            continue
        # Count complete byte strings first: constructing one Python object per
        # byte for every occurrence dominates preprocessing on a real corpus.
        raw_counts.update(match.group().encode('utf-8') for match in PATTERN.finditer(chunk))
    del text, chunks
    counts = {tuple(bytes([b]) for b in word): count for word, count in raw_counts.items()}
    del raw_counts
    words, freqs = list(counts), list(counts.values())
    pairs = Counter()
    owners = defaultdict(set)
    for wid, word in enumerate(words):
        for pair, count in Counter(zip(word, word[1:])).items():
            pairs[pair] += count * freqs[wid]
            owners[pair].add(wid)
    heap = [(-count, ReversePair(pair)) for pair, count in pairs.items()]
    heapq.heapify(heap)
    merges = []
    while len(vocab) < vocab_size and heap:
        neg, item = heapq.heappop(heap)
        best = item.pair
        if -neg != pairs[best] or pairs[best] <= 0:
            continue
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        touched = set()
        for wid in list(owners[best]):
            old = words[wid]
            new = merge_word(old, best)
            for pair, count in Counter(zip(old, old[1:])).items():
                pairs[pair] -= count * freqs[wid]
                owners[pair].discard(wid)
                touched.add(pair)
            words[wid] = new
            for pair, count in Counter(zip(new, new[1:])).items():
                pairs[pair] += count * freqs[wid]
                owners[pair].add(wid)
                touched.add(pair)
        for pair in touched:
            if pairs[pair] > 0:
                heapq.heappush(heap, (-pairs[pair], ReversePair(pair)))
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.inverse = {v: k for k, v in self.vocab.items()}
        for token in self.special_tokens:
            value = token.encode('utf-8')
            if value not in self.inverse:
                idx = max(self.vocab, default=-1) + 1
                self.vocab[idx] = value
                self.inverse[value] = idx
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.splitter = special_pattern(self.special_tokens)
        self.special_set = set(self.special_tokens)

    @lru_cache(maxsize=8192)
    def _encode_word(self, value):
        word = tuple(bytes([b]) for b in value)
        while len(word) > 1:
            candidates = [(self.ranks[pair], pair) for pair in zip(word, word[1:]) if pair in self.ranks]
            if not candidates:
                break
            _, best = min(candidates, key=lambda p: p[0])
            word = merge_word(word, best)
        return tuple(self.inverse[piece] for piece in word)

    def encode(self, text):
        ids = []
        for chunk in self.splitter.split(text) if self.splitter else [text]:
            if chunk in self.special_set:
                ids.append(self.inverse[chunk.encode('utf-8')])
            else:
                for match in PATTERN.finditer(chunk):
                    ids.extend(self._encode_word(match.group().encode('utf-8')))
        return ids

    def encode_iterable(self, iterable):
        # Each item is a complete text segment (e.g. a document or line),
        # matching the assignment's iterator contract. No entire corpus buffering.
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

    def save(self, path):
        payload = {'format':'cs336-byte-bpe-v1', 'vocab':{str(k):base64.b64encode(v).decode('ascii') for k,v in self.vocab.items()},
                   'merges':[[base64.b64encode(a).decode('ascii'),base64.b64encode(b).decode('ascii')] for a,b in self.merges], 'special_tokens':self.special_tokens}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls({int(k):base64.b64decode(v) for k,v in obj['vocab'].items()},
                   [tuple(base64.b64decode(x) for x in pair) for pair in obj['merges']], obj['special_tokens'])

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # GPT-2's reversible byte-to-visible-character vocabulary representation.
        bs = list(range(ord('!'), ord('~')+1)) + list(range(161,173)) + list(range(174,256))
        cs, n = bs.copy(), 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256+n); n += 1
        decoder = {chr(c):b for b,c in zip(bs,cs)}
        raw = json.loads(Path(vocab_filepath).read_text(encoding='utf-8'))
        vocab = {idx:bytes(decoder[c] for c in token) for token,idx in raw.items()}
        merges = []
        for line in Path(merges_filepath).read_text(encoding='utf-8').splitlines():
            if not line or line.startswith('#'):
                continue
            a,b = line.split(' ')
            merges.append((bytes(decoder[c] for c in a),bytes(decoder[c] for c in b)))
        return cls(vocab, merges, special_tokens)
