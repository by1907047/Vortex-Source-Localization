from pathlib import Path
import sys,json
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import experiment
experiment.SOURCE=ROOT/'data'
mapping=json.loads((ROOT/'results'/'seed_mapping.json').read_text())['run_id_to_actual_seed']
run_ids=sorted(map(int,mapping))
seeds=[mapping[str(r)] for r in run_ids]
out=ROOT/'reproduced_results'
if out.exists():raise RuntimeError('reproduced_results already exists; choose a new output path')
out.mkdir()
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
rows=[]
for task in ['cylinder','fish']:rows.extend(experiment.train_task(task,out,seeds,500,32,run_ids))
import pandas as pd
frame=pd.DataFrame(rows)
frame.to_csv(out/'run_metrics.csv',index=False)
frame.groupby(['task','config','region','metric'],sort=False).error_cm.agg(['count','mean','std']).reset_index().to_csv(out/'summary.csv',index=False)
