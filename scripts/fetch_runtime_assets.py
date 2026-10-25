"""Fetch only tokenizer/prompt runtime assets; verify content before use."""
import hashlib,json,os,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for row in json.loads((root/'provenance/runtime-assets.json').read_text())['files']:
    target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']:continue
    with urllib.request.urlopen(row['url'],timeout=90) as response:raw=response.read()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Runtime asset SHA256 mismatch: '+row['path'])
    temporary=target.with_suffix(target.suffix+'.partial');temporary.write_bytes(raw);os.replace(temporary,target)
    print('VERIFIED',row['path'])
