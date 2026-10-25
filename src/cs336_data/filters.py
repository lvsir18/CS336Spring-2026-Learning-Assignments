"""HTML extraction, externally trained classifiers, and transparent PII/rule filters."""
from functools import lru_cache
import ipaddress
import os
from pathlib import Path
import re


def extract_text_from_html_bytes(html_bytes):
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding
    try:
        text = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        text = html_bytes.decode(detect_encoding(html_bytes) or 'windows-1252',errors='replace')
    return extract_plain_text(text)


@lru_cache(maxsize=8)
def _model(path):
    import fasttext
    if not Path(path).is_file():
        raise FileNotFoundError(f'Classifier not found: {path}. See docs/a4-data.md for acquisition/training.')
    return fasttext.load_model(path)


def _predict(text,variable,default):
    path = os.getenv(variable,str(Path('models')/default))
    labels,probabilities = _model(path).predict(' '.join(text.split()),k=1)
    return labels[0].removeprefix('__label__'),float(probabilities[0])


def identify_language(text):
    return _predict(text,'CS336_LID_MODEL','lid.176.bin')
def classify_nsfw(text):
    return _predict(text,'CS336_NSFW_MODEL','dolma_fasttext_nsfw_jigsaw_model.bin')
def classify_toxic_speech(text):
    return _predict(text,'CS336_TOXIC_MODEL','dolma_fasttext_hatespeech_jigsaw_model.bin')
def classify_quality(text):
    return _predict(text,'CS336_QUALITY_MODEL','quality.bin')


def mask_emails(text):
    return re.subn(r'(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+', '|||EMAIL_ADDRESS|||',text)


def mask_phone_numbers(text):
    pattern = r'(?<!\w)(?:\+?1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-]?)\d{3}[ .-]?\d{4}(?!\w)'
    return re.subn(pattern,'|||PHONE_NUMBER|||',text)


def mask_ips(text):
    count = 0
    def replace(match):
        nonlocal count
        try:
            ipaddress.IPv4Address(match.group())
        except ipaddress.AddressValueError:
            return match.group()
        count += 1
        return '|||IP_ADDRESS|||'
    masked = re.sub(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)',replace,text)
    return masked,count


def gopher_quality_filter(text):
    words = text.split()
    if not 50 <= len(words) <= 100000:
        return False
    mean_length = sum(map(len,words))/len(words)
    if not 3 <= mean_length <= 10:
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and sum(line.rstrip().endswith(('...','…')) for line in lines)/len(lines) > .3:
        return False
    return sum(any(c.isalpha() for c in word) for word in words)/len(words) >= .8
