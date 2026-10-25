"""Explicit, verified public downloads for the A2-A5 single-GPU experiment suite."""
import argparse,concurrent.futures,gzip,hashlib,json,os,subprocess,time
from pathlib import Path
from cs336_basics.training import atomic_json,run_lock


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def metadata(url):
    return json.loads(subprocess.check_output(['curl','-fLsS','--retry','3','--connect-timeout','20','--max-time','90',url]))


def fetch(url,path,sha=None,blob=None,size=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def valid(p):
        if not p.exists() or (size is not None and p.stat().st_size!=size):return False
        if sha:return sha256(p)==sha
        if blob:
            raw=p.read_bytes();return hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==blob
        return True
    if not valid(path):
        part=path.with_name(path.name+'.partial')
        subprocess.run(['curl','-fLsS','--retry','5','--retry-all-errors','--connect-timeout','20','--max-time','14400',
                        '-C','-','-o',str(part),url],check=True)
        if not valid(part):raise ValueError(f'Hash/size verification failed: {path}')
        os.replace(part,path)
    return {'path':str(path),'url':url,'bytes':path.stat().st_size,'sha256':sha256(path)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoint',default='https://hf-mirror.com');a=p.parse_args()
    pins=json.loads(Path('provenance/hf-revisions.json').read_text()); root=Path('data/formal-assets');root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        (root/'downloader.pid').write_text(str(os.getpid())+'\n')
        tasks=[];repos={}
        for repo,destination,selected in [
            ('allenai/OLMo-2-0425-1B','models/olmo-2-0425-1b',None),
            ('allenai/dolma-jigsaw-fasttext-bigrams-nsfw','models/formal/nsfw',{'model.bin'}),
            ('allenai/dolma-jigsaw-fasttext-bigrams-hatespeech','models/formal/toxic',{'model.bin'})]:
            info=metadata(f'{a.endpoint}/api/models/{repo}/revision/{pins[repo]}?blobs=true');repos[repo]=info['sha']
            for item in info['siblings']:
                name=item['rfilename']
                wanted=name in selected if selected is not None else name.endswith(('.json','.safetensors','.txt','.jinja','.md')) or name.startswith('LICENSE')
                if not wanted:continue
                lfs=item.get('lfs',{})
                tasks.append((f'{a.endpoint}/{repo}/resolve/{pins[repo]}/{name}',Path(destination)/name,lfs.get('sha256'),None if lfs else item.get('blobId'),item.get('size')))
        for repo,subdir,destination,selected in [
            ('openai/gsm8k','main','data/formal-assets/gsm8k',None),
            ('Salesforce/wikitext','wikitext-2-raw-v1','data/formal-assets/wiki',{'train-00000-of-00001.parquet'})]:
            info=metadata(f'{a.endpoint}/api/datasets/{repo}/revision/{pins[repo]}');repos[repo]=info['sha']
            tree=metadata(f'{a.endpoint}/api/datasets/{repo}/tree/{pins[repo]}/{subdir}')
            for item in tree:
                name=Path(item['path']).name
                if item['type']!='file' or (selected and name not in selected):continue
                tasks.append((f'{a.endpoint}/datasets/{repo}/resolve/{pins[repo]}/{item["path"]}',Path(destination)/name,item.get('lfs',{}).get('oid'),None,item.get('size')))
        tasks.append(('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin',Path('models/formal/lid.176.bin'),None,None,None))
        for kind in ['wet','warc']:
            paths=root/f'{kind}.paths.gz'
            fetch(f'https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/{kind}.paths.gz',paths)
            first=gzip.decompress(paths.read_bytes()).decode().splitlines()[0]
            tasks.append(('https://data.commoncrawl.org/'+first,root/f'example.{kind}.gz',None,None,None))
        completed=[]
        def one(task):
            row=fetch(*task);print('DOWNLOADED',json.dumps(row),flush=True);return row
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(one,tasks):
                completed.append(row)
                atomic_json(root/'download-progress.json',{'completed':completed,'repositories':repos,'time_unix':time.time()})
        atomic_json(root/'manifest.json',{'files':completed,'repositories':repos,'time_unix':time.time(),
                    'note':'HF LFS files verified by published SHA256; Git blobs by blob SHA1; Common Crawl/lid hashes recorded on first download'})
        print('ALL_ASSETS_READY',flush=True)


if __name__=='__main__':main()
