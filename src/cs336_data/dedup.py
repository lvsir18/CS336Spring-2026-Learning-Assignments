"""Exact line deduplication and seeded MinHash+LSH+Jaccard verification."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import random
import re
import unicodedata


def _paths(input_files,output_directory):
    files = [Path(p) for p in input_files]
    if len({p.name for p in files}) != len(files):
        raise ValueError('Input basenames must be unique to avoid output overwrite')
    output = Path(output_directory); output.mkdir(parents=True,exist_ok=True)
    if any(p.resolve().parent == output.resolve() for p in files):
        raise ValueError('Output directory must differ from the input directory')
    return files,output


def exact_line_deduplication(input_files,output_directory):
    files,output = _paths(input_files,output_directory)
    counts = Counter()
    for path in files:
        with path.open(encoding='utf-8') as f:
            counts.update(line.rstrip('\r\n') for line in f)
    for path in files:
        with path.open(encoding='utf-8') as source, (output/path.name).open('w',encoding='utf-8',newline='') as dest:
            for line in source:
                if counts[line.rstrip('\r\n')] == 1:
                    dest.write(line)


def normalized_words(text):
    text = unicodedata.normalize('NFD',text.lower())
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    return text.split()


def shingles(text,ngrams):
    words = normalized_words(text)
    if len(words) < ngrams:
        # Distinct short documents must not collapse to the same empty set.
        return {tuple(words)}
    return {tuple(words[i:i+ngrams]) for i in range(len(words)-ngrams+1)}


def minhash_deduplication(input_files,num_hashes,num_bands,ngrams,jaccard_threshold,output_directory):
    if num_hashes <= 0 or num_bands <= 0 or num_hashes%num_bands or ngrams <= 0 or not 0 <= jaccard_threshold <= 1:
        raise ValueError('Invalid MinHash/LSH settings')
    files,output = _paths(input_files,output_directory)
    rng = random.Random(42)
    prime = 2**61-1
    coefficients = [(rng.randrange(1,prime),rng.randrange(prime)) for _ in range(num_hashes)]
    bands = defaultdict(list)
    parent = list(range(len(files)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        parent[max(a,b)]=min(a,b)
    sets = []
    for idx,path in enumerate(files):
        tokens = shingles(path.read_text(encoding='utf-8'),ngrams)
        sets.append(tokens)
        values = [int.from_bytes(hashlib.blake2b('\0'.join(s).encode('utf-8'),digest_size=8).digest(),'big')%prime for s in tokens]
        signature = [min((a*x+b)%prime for x in values) for a,b in coefficients]
        candidates = set()
        width = num_hashes//num_bands
        for band in range(num_bands):
            key = (band,tuple(signature[band*width:(band+1)*width]))
            candidates.update(bands[key]); bands[key].append(idx)
        for candidate in candidates:
            other = sets[candidate]
            similarity = len(tokens & other)/len(tokens | other)
            if similarity >= jaccard_threshold:
                union(idx,candidate)
    for idx,path in enumerate(files):
        if find(idx)==idx:
            (output/path.name).write_bytes(path.read_bytes())
    return {'input_documents':len(files),'retained_documents':sum(find(i)==i for i in range(len(files))), 'seed':42}
