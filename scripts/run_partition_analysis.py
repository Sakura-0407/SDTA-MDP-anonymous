"""Reconstruct the manuscript's noise-aware symbolic partitions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sdta_mdp.benchmarks import STOCHASTIC_MAIN_BENCHMARKS, make_benchmark
from sdta_mdp.evaluation import _configure_stochastic_noise
from sdta_mdp.symbolic import SymbolicPartitioner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmarks',nargs='+',choices=STOCHASTIC_MAIN_BENCHMARKS,default=list(STOCHASTIC_MAIN_BENCHMARKS))
    p.add_argument('--out-dir',type=Path,default=ROOT/'runs/partition')
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in args.benchmarks:
        env=make_benchmark(task)
        _configure_stochastic_noise(env,slip_probability=.1,continuous_noise_fraction=.05,
                                    project_stochastic_paths=True,gaussian_support_sigma=2.,gaussian_support_points=3)
        start=perf_counter()
        blocks,counts=SymbolicPartitioner(env,frontier_mode='sampled',seed=0).build_partition()
        row={'benchmark':task,'blocks':len(blocks),'path_signature_counts':dict(counts),
             'partition_seconds':perf_counter()-start,'dimensions':len(env.state_names),
             'finegrid_cells':8**len(env.state_names),'tile_cells':4**len(env.state_names),'random_prototypes':16}
        rows.append(row);print(task,len(blocks),flush=True)
    (args.out_dir/'partitions.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    # The sampled frontier/absorption counts require the full main experiment;
    # they are not inferred from this independent partition-only reconstruction.


if __name__=='__main__':main()
