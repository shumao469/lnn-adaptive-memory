"""Plot paired, multiplicity-adjusted Reacher gate contrasts from reported data."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'nc_memory_reacher_corrected_20261125' / 'corrected_analysis.json'
payload = json.loads(source.read_text(encoding='utf-8-sig'))
rows = payload['contrasts']
assert len(rows) == 48

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#263646', 'xtick.color': '#425568',
                     'ytick.color': '#425568', 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), sharey=True,
                         gridspec_kw={'hspace': .65, 'wspace': .26})
colors = {'q_up': '#2B6A99', 'r_up': '#C76847'}
conditions = [('q_up',0),('q_up',40),('q_up',80),
              ('r_up',0),('r_up',40),('r_up',80)]
ys = np.array([5,4,3,2,1,0])
for i, comp in enumerate(('innovation','huber_kf')):
    for j, metric in enumerate(('tracking_rmse','control_energy')):
        ax=axes[i,j]
        ax.axvline(0, color='#8A9BA9', lw=1, ls='--', zorder=0)
        for y,(reg,delay) in zip(ys,conditions):
            m=[z for z in rows if z['comparator']==comp and z['metric']==metric
               and z['regime']==reg and z['delay_ms']==delay]
            assert len(m)==1
            r=m[0]; factor=1000 if metric=='tracking_rmse' else 1
            low,high=np.array(r['bonferroni_familywise_95'])*factor
            x=r['estimate']*factor
            ax.hlines(y,low,high,color=colors[reg],lw=2.2,zorder=2)
            ax.plot(x,y,'o',ms=5.5,color=colors[reg],zorder=3)
        ax.set_yticks(ys,[f'{label}  /  {delay} ms' for label,delay in
                           [('Q increase',0),('Q increase',40),('Q increase',80),
                            ('R increase',0),('R increase',40),('R increase',80)]])
        ax.axhline(2.5,lw=.8,color='#E1E8ED')
        ax.grid(axis='x',color='#E9EEF2',lw=.8)
        ax.set_ylim(-.7,5.7)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y',length=0)
        if j==1: ax.tick_params(axis='y',labelleft=False)
        ax.set_title(('Tracking error' if j==0 else 'Squared-torque integral'),
                     loc='left',weight='bold',fontsize=11,pad=8)
        ax.set_xlabel('Learned minus comparator  •  '+('mm' if j==0 else 'squared-torque integral'))
fig.suptitle('Memory-write gating in simulated Reacher', x=.075, y=.98,
             ha='left',fontsize=17,weight='bold',color='#132B3C')
fig.text(.075,.925,'Exploratory paired comparisons  |  lower values favour the learned gate',
         ha='left',fontsize=10.5,color='#526B7B')
fig.text(.175,.86,'A  Learned gate vs innovation threshold',weight='bold',fontsize=12,color='#1E3346')
fig.text(.175,.426,'B  Learned gate vs Huber Kalman',weight='bold',fontsize=12,color='#1E3346')
fig.text(.075,.018,'Points: paired means; bars: 95% Bonferroni-adjusted intervals across 48 post hoc comparisons.  '
         '24 scenarios × 3 fits per condition; 100,000 crossed bootstrap draws.\n'
         'Simulation uses a known post-switch horizon and equal write quotas; Huber trajectories are reused across fits. '
         'Intervals are approximate.',fontsize=8.5,color='#536976',ha='left',va='bottom')
fig.subplots_adjust(left=.175,right=.965,top=.81,bottom=.105)
target = Path(__file__).with_name('reacher_gate_results.png')
fig.savefig(target,dpi=210,bbox_inches='tight')
print(target)
