import json
from pathlib import Path
r=Path('runs/formal/remaining');results=json.loads((r/'results.json').read_text())
lines=['# A2–A5 单卡实验结果快照','','本文件由真实产物生成。范围和局限见 plan.json。','','|任务|状态|退出码|秒数|','|---|---|---:|---:|']
for name,row in results.items():lines.append(f'|{name}|{row["status"]}|{row.get("exit_code", "")}|{row.get("elapsed_seconds",0):.1f}|')
for name in ['a4-raw','a4-clean']:
    p=r/name/'metrics.jsonl'
    if p.exists():
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        lines+=['',f'## {name}','','```json',json.dumps(rows[-1],ensure_ascii=False,indent=2),'```']
for name in ['a5-base-test','a5-sft-test','a5-rl-test','a5-dpo-test']:
    p=r/name/'eval.summary.json'
    if p.exists():lines+=['',f'## {name}','','```json',p.read_text(),'```']
(r/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
