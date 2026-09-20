"""Regenerate manuscript statistics from the curated seven-task records."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sdta_mdp.evaluation import _bootstrap_ci, _paired_sign_flip_pvalue

TASKS = ('stochastic_braking_car','stochastic_point_mass','stochastic_double_integrator_parking',
         'stochastic_continuous_braking_car','stochastic_pendulum_swing_up',
         'stochastic_continuous_point_mass','stochastic_continuous_double_integrator_parking')
LABELS = dict(zip(TASKS,('BrakingCar','PointMass','DoubleIntegrator','Cont. BrakingCar',
                        'Pendulum','Cont. PointMass','Cont. DoubleIntegrator')))


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['benchmark'],row['method']].append(row)
    result=[]
    for (benchmark,method),group in sorted(grouped.items()):
        seeds=[int(r['seed']) for r in group]
        if len(seeds)!=len(set(seeds)):raise ValueError('Duplicate task/method/seed')
        item={'benchmark':benchmark,'method':method,'seeds':len(seeds)}
        for metric in ['mean_cost','success_rate','violation_rate','model_calls','seconds','preparation_seconds','evaluation_seconds']:
            if not all(r.get(metric,'')!='' for r in group):continue
            values=[float(r[metric]) for r in group]
            low,high=_bootstrap_ci(values)
            item.update({metric:mean(values),metric+'_ci_low':low,metric+'_ci_high':high})
        result.append(item)
    # Different methods omit some diagnostics; use a common CSV schema.
    keys=list(dict.fromkeys(k for item in result for k in item))
    return [{k:item.get(k,'') for k in keys} for item in result]


def scaling(partition):
    result=[]
    for regime in ['finite-action','continuous-action']:
        group=[r for r in partition if r['action_regime']==regime]
        for bins in [8,16,32,64,128]:
            cells=bins**2;tile=(bins//2)**2
            blocks=[float(r['sdta_blocks_mean']) for r in group]
            result.append({'action_regime':regime,'benchmarks':len(group),'bins_per_dimension':bins,
                'finegrid_cells':cells,'tile_cells':tile,'mean_sdta_blocks':mean(blocks),
                'mean_grid_to_sdta':mean(cells/b for b in blocks),
                'mean_tile_to_sdta':mean(tile/b for b in blocks)})
    return result


def plot(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index={(r['benchmark'],r['method']):r for r in summary}
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.2),layout='constrained')
    markers=['o','s','^','D','v','P','X']
    for axis,metric,title in zip(axes,['mean_cost','model_calls'],['Control cost','Complete model evaluations']):
        values=[]
        for task,marker in zip(TASKS,markers):
            x=float(index[task,'MPC'][metric]);y=float(index[task,'SDTA-MDP'][metric])
            axis.scatter(x,y,label=LABELS[task],marker=marker,s=34)
            values.extend([x,y])
        low,high=min(values)*.7,max(values)*1.4
        axis.plot([low,high],[low,high],color='0.5',linestyle='--',linewidth=.7)
        axis.set(xscale='log',yscale='log',xlabel='MPC',ylabel='SDTA-MDP',title=title,
                 xlim=(low,high),ylim=(low,high))
        axis.grid(alpha=.15)
    axes[1].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1.02,1))
    fig.savefig(path);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,default=ROOT/'reference')
    p.add_argument('--out-dir',type=Path,default=ROOT/'generated')
    p.add_argument('--figure',action='store_true')
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    rows=read(args.reference_dir/'main.csv')
    expected={(b,m,s) for b in TASKS for m in (['SDTA-MDP','MPC','FineGridVI','TileQ']+(['SymPar-Q-Upstream'] if b in TASKS[:3] else [])) for s in range(10)}
    observed={(r['benchmark'],r['method'],int(r['seed'])) for r in rows}
    if observed!=expected or len(rows)!=len(expected):raise ValueError('The reference main study must contain exactly 310 unique records.')
    summary=summarize(rows);write(args.out_dir/'main_summary.csv',summary)
    write(args.out_dir/'lookahead_summary.csv',summarize(read(args.reference_dir/'lookahead_ablation.csv')))
    write(args.out_dir/'partition_scaling.csv',scaling(read(args.reference_dir/'partition_sizes.csv')))
    paired=[]
    index={(r['benchmark'],r['method'],int(r['seed'])):r for r in rows}
    for task in TASKS:
        for metric in ['mean_cost','success_rate','violation_rate','model_calls']:
            delta=[float(index[task,'SDTA-MDP',s][metric])-float(index[task,'MPC',s][metric]) for s in range(10)]
            paired.append({'benchmark':task,'metric':metric,'mean_sdta_minus_mpc':mean(delta),
                           'paired_sign_flip_pvalue':_paired_sign_flip_pvalue(delta)})
    write(args.out_dir/'paired_sdta_mpc.csv',paired)
    if args.figure:plot(summary,args.out_dir/'experiment_key_results.pdf')
    print(f'Summarized {len(rows)} main records and 80 ablation records in {args.out_dir}')


if __name__=='__main__':main()
