"""
Large-Scale Prospective Validation v2
======================================
150 new configs × 4 new seeds, SAME threshold (0.0505) locked from calibration.
Score range: 0.029–0.370, spanning LOW (<0.062) and HIGH (>0.13) bands.
No overlap with original 50-config experiment in parameters OR seeds.
Ground truth: mean_delta > 0.03 (3pp, same as original experiment).
"""
import sys, os, json, time
sys.path.insert(0,'.')

import numpy as np
from experiments.predictor_prospective import (
    make_graph, graph_stats, run_config, OUTCOME_MARGIN
)

THRESHOLD   = 0.0505   # LOCKED
NEW_SEEDS   = [11, 22, 33, 44, 55, 66, 77, 88]   # not used in original
CHECKPOINT  = 'results/large_scale_v2_checkpoint.json'
CHUNK_SIZE  = 4

# ── Config generation ────────────────────────────────────────────────────────

def build_configs():
    from experiments.predictor_prospective import CALIB_LOW, CALIB_HIGH, HELD_LOW, HELD_HIGH
    orig_low  = set((c[0],c[1],c[2],c[3]) for c in CALIB_LOW+HELD_LOW)
    orig_high = set((c[0],c[1],c[2],c[3]) for c in CALIB_HIGH+HELD_HIGH)

    low_configs=[]; high_configs=[]
    for p_in in [0.010,0.011,0.012,0.013,0.014,0.015,0.016,0.017]:
        for p_out in [0.016,0.018,0.020,0.022,0.024,0.026]:
            for hub,extra in [(0.00,0),(0.01,6),(0.02,10)]:
                k=(p_in,p_out,hub,extra)
                if k in orig_low: continue
                ei,x,y,n,nc=make_graph(p_in,p_out,hub,extra,seed=11)
                h,cv,score=graph_stats(ei,y,n)
                if score<0.062:
                    low_configs.append({'p_intra':p_in,'p_inter':p_out,'hub_pct':hub,'extra':extra,'score':round(score,5),'band':'LOW'})

    for p_in in [0.038,0.042,0.047,0.053,0.058,0.064,0.069,0.075,0.081,0.087,0.093]:
        for p_out in [0.008,0.009,0.010,0.011,0.012,0.013]:
            for hub,extra in [(0.02,14),(0.04,24),(0.06,34),(0.08,44)]:
                k=(p_in,p_out,hub,extra)
                if k in orig_high: continue
                ei,x,y,n,nc=make_graph(p_in,p_out,hub,extra,seed=11)
                h,cv,score=graph_stats(ei,y,n)
                if score>0.13:
                    high_configs.append({'p_intra':p_in,'p_inter':p_out,'hub_pct':hub,'extra':extra,'score':round(score,5),'band':'HIGH'})

    low_configs.sort(key=lambda x: x['score'])
    high_configs.sort(key=lambda x: x['score'])
    step_l=max(1,len(low_configs)//75)
    step_h=max(1,len(high_configs)//75)
    return low_configs[::step_l][:75] + high_configs[::step_h][:75]


# ── Metrics ──────────────────────────────────────────────────────────────────

def cm(results, margin=OUTCOME_MARGIN):
    TP=FP=FN=TN=0
    for r in results:
        pred=r['score']>THRESHOLD; actual=r['mean_delta']>margin
        if pred and actual: TP+=1
        elif pred and not actual: FP+=1
        elif not pred and actual: FN+=1
        else: TN+=1
    n=TP+FP+FN+TN; acc=(TP+TN)/n
    prec=TP/(TP+FP) if TP+FP else float('nan')
    rec=TP/(TP+FN) if TP+FN else float('nan')
    spec=TN/(TN+FP) if TN+FP else float('nan')
    bal=(rec+spec)/2 if not(np.isnan(rec) or np.isnan(spec)) else float('nan')
    den=((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    mcc=(TP*TN-FP*FN)/den if den else float('nan')
    return dict(TP=TP,FP=FP,FN=FN,TN=TN,n=n,acc=acc,prec=prec,rec=rec,spec=spec,bal=bal,mcc=mcc)


# ── Runner ───────────────────────────────────────────────────────────────────

def run_next_chunk():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: state=json.load(f)
    else:
        state={'configs':build_configs(),'results':[]}
        with open(CHECKPOINT,'w') as f: json.dump(state,f,indent=2)
        cfgs=state['configs']
        low=[c for c in cfgs if c['band']=='LOW']
        high=[c for c in cfgs if c['band']=='HIGH']
        print(f'Generated {len(cfgs)} configs: {len(low)} LOW + {len(high)} HIGH')

    configs=state['configs']
    done_ids=set(r['config_id'] for r in state['results'])
    todo=[(i,c) for i,c in enumerate(configs) if i not in done_ids]
    total=len(configs)

    if not todo:
        print('All done.'); return True

    t0=time.time()
    for i,(idx,cfg) in enumerate(todo[:CHUNK_SIZE]):
        r=run_config(cfg['p_intra'],cfg['p_inter'],cfg['hub_pct'],cfg['extra'],seeds=NEW_SEEDS)
        r['config_id']=idx; r['band']=cfg['band']
        state['results'].append(r)
        done_now=len(state['results']); remaining=total-done_now
        eta=(time.time()-t0)/(i+1)*remaining
        correct=(r['score']>THRESHOLD)==(r['mean_delta']>OUTCOME_MARGIN)
        print(f"  [{done_now:3d}/{total}] score={r['score']:.4f} band={cfg['band']:4s} "
              f"Δ={r['mean_delta']:+.4f} {'SUBST' if r['mean_delta']>OUTCOME_MARGIN else 'NEGL ':5s} "
              f"pred={'✓' if correct else '✗'} ETA={eta:.0f}s")
        with open(CHECKPOINT,'w') as f: json.dump(state,f,indent=2)

    remaining=total-len(state['results'])
    if remaining>0:
        elapsed=time.time()-t0
        print(f'\nChunk done. {remaining} remain (~{elapsed/CHUNK_SIZE*remaining/60:.0f} min).')
        return False
    return True


def finalise():
    with open(CHECKPOINT) as f: state=json.load(f)
    results=state['results']
    from scipy import stats as scipy_stats

    scores=np.array([r['score'] for r in results])
    deltas=np.array([r['mean_delta'] for r in results])
    r_val,p_val=scipy_stats.pearsonr(scores,deltas)

    m=cm(results)
    low_r=[r for r in results if r['band']=='LOW']
    high_r=[r for r in results if r['band']=='HIGH']

    print('\n'+'='*68)
    print('LARGE-SCALE PROSPECTIVE VALIDATION v2')
    print(f'N={len(results)} graphs | Threshold={THRESHOLD} LOCKED | Margin={OUTCOME_MARGIN*100:.0f}pp')
    print('='*68)
    print(f'\nCorrelation (score → delta): r={r_val:+.3f}  p={p_val:.2e}')
    print(f'\nConfusion Matrix:')
    print(f"  {'':22s}  Actual SUBST  Actual NEGL")
    print(f"  {'Predict SUBST':22s}  {m['TP']:>5} (TP)   {m['FP']:>5} (FP)")
    print(f"  {'Predict NEGL':22s}  {m['FN']:>5} (FN)   {m['TN']:>5} (TN)")
    print(f"\n  Accuracy:         {m['acc']:.1%}  ({m['TP']+m['TN']}/{m['n']})")
    print(f"  Balanced Acc:     {m['bal']:.1%}")
    print(f"  MCC:              {m['mcc']:+.3f}")
    print(f"  Precision:        {m['prec']:.1%}")
    print(f"  Recall:           {m['rec']:.1%}")
    print(f"  Specificity:      {m['spec']:.1%}")

    pos=[r for r in results if r['score']>THRESHOLD]
    neg=[r for r in results if r['score']<=THRESHOLD]
    if pos: print(f'\n  Rule-positive (n={len(pos)}): mean Δ={np.mean([r["mean_delta"] for r in pos])*100:+.2f}pp')
    if neg: print(f'  Rule-negative (n={len(neg)}): mean Δ={np.mean([r["mean_delta"] for r in neg])*100:+.2f}pp')

    os.makedirs('results',exist_ok=True)
    out={'threshold':THRESHOLD,'outcome_margin_pp':OUTCOME_MARGIN*100,
         'n':len(results),'correlation_r':float(r_val),'correlation_p':float(p_val),
         'metrics':m,'results':results}
    with open('results/large_scale_v2.json','w') as f: json.dump(out,f,indent=2)
    print('\nSaved → results/large_scale_v2.json')
    return out


if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser()
    p.add_argument('--finalise',action='store_true')
    args=p.parse_args()
    if args.finalise: finalise()
    else:
        done=run_next_chunk()
        if done: finalise()
