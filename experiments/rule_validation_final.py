"""
experiments/rule_validation_final.py

Final Prospective Rule Validation
==================================

RULE (frozen, never tuned after initial calibration):
    SCORE(G) = homophily(G) × deg_cv(G)
    Predict "TGS wins" iff SCORE > 0.0505

This is the definitive prospective test:
  - 74 completely new graphs (no parameter or seed overlap with any prior experiment)
  - 8 seeds per graph (same as large_scale_v2)
  - Reports: accuracy, MCC, precision, recall, specificity, balanced accuracy,
    calibration curve, and annotated FP/FN examples

Seeds: [111, 222, 333, 444, 555, 666, 777, 888] — not used anywhere before
Config overlap: zero (verified against all prior checkpoints)

Combined with large_scale_v2 (150 graphs):
  Total prospective evaluations: 224 graphs
  Threshold never adjusted after original 20-graph calibration.
"""

import sys, os, json, time
sys.path.insert(0, ".")

import numpy as np
from experiments.predictor_prospective import make_graph, graph_stats, run_config, OUTCOME_MARGIN

THRESHOLD  = 0.0505   # LOCKED — never changes
NEW_SEEDS  = [111, 222, 333, 444, 555, 666, 777, 888]
CHECKPOINT = "results/rule_validation_checkpoint.json"
CHUNK_SIZE = 5

LOW_CONFIGS = [
    (0.010,0.017,0.00,0),(0.010,0.019,0.00,0),(0.010,0.021,0.00,0),(0.010,0.023,0.00,0),(0.010,0.025,0.00,0),
    (0.011,0.017,0.00,0),(0.011,0.019,0.00,0),(0.011,0.021,0.00,0),(0.011,0.023,0.00,0),(0.011,0.025,0.00,0),
    (0.012,0.017,0.00,0),(0.012,0.019,0.00,0),(0.012,0.021,0.00,0),(0.012,0.023,0.00,0),(0.012,0.025,0.00,0),
    (0.013,0.017,0.00,0),(0.013,0.019,0.00,0),(0.013,0.021,0.00,0),(0.013,0.023,0.00,0),
    (0.014,0.017,0.00,0),(0.014,0.019,0.00,0),(0.014,0.021,0.00,0),(0.014,0.023,0.00,0),
    (0.010,0.019,0.01,6),(0.011,0.019,0.01,6),(0.012,0.019,0.01,6),(0.013,0.019,0.01,6),
    (0.010,0.021,0.01,6),(0.011,0.021,0.01,6),(0.012,0.021,0.01,6),(0.013,0.021,0.01,6),
    (0.010,0.023,0.01,6),(0.011,0.023,0.01,6),(0.012,0.023,0.01,6),
    (0.015,0.017,0.00,0),(0.015,0.019,0.00,0),(0.015,0.021,0.00,0),
]

HIGH_CONFIGS = [
    (0.036,0.010,0.02,13),(0.036,0.009,0.04,23),(0.036,0.008,0.06,33),
    (0.041,0.010,0.02,13),(0.041,0.009,0.04,23),(0.041,0.008,0.06,33),(0.041,0.008,0.08,43),
    (0.046,0.011,0.02,13),(0.046,0.010,0.04,23),(0.046,0.009,0.06,33),(0.046,0.008,0.08,43),
    (0.051,0.011,0.02,13),(0.051,0.010,0.04,23),(0.051,0.009,0.06,33),(0.051,0.008,0.08,43),
    (0.056,0.011,0.02,13),(0.056,0.010,0.04,23),(0.056,0.009,0.06,33),(0.056,0.008,0.08,43),
    (0.061,0.011,0.02,13),(0.061,0.010,0.04,23),(0.061,0.009,0.06,33),(0.061,0.008,0.08,43),
    (0.066,0.010,0.04,23),(0.066,0.009,0.06,33),(0.066,0.008,0.08,43),
    (0.072,0.010,0.04,23),(0.072,0.009,0.06,33),(0.072,0.008,0.08,43),
    (0.078,0.009,0.06,33),(0.078,0.008,0.08,43),
    (0.083,0.009,0.06,33),(0.083,0.008,0.08,43),
    (0.089,0.009,0.06,33),(0.089,0.008,0.08,43),
    (0.051,0.012,0.02,13),(0.056,0.012,0.04,23),
]

ALL_CONFIGS = [("LOW",c) for c in LOW_CONFIGS] + [("HIGH",c) for c in HIGH_CONFIGS]


def cm_metrics(results, margin=OUTCOME_MARGIN):
    TP=FP=FN=TN=0
    for r in results:
        pred=r["score"]>THRESHOLD; actual=r["mean_delta"]>margin
        if pred and actual: TP+=1
        elif pred and not actual: FP+=1
        elif not pred and actual: FN+=1
        else: TN+=1
    n=TP+FP+FN+TN; acc=(TP+TN)/n
    prec=TP/(TP+FP) if TP+FP else float("nan")
    rec=TP/(TP+FN) if TP+FN else float("nan")
    spec=TN/(TN+FP) if TN+FP else float("nan")
    bal=(rec+spec)/2 if not(np.isnan(rec) or np.isnan(spec)) else float("nan")
    den=((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5
    mcc=(TP*TN-FP*FN)/den if den else float("nan")
    return dict(TP=TP,FP=FP,FN=FN,TN=TN,n=n,acc=acc,prec=prec,rec=rec,spec=spec,bal=bal,mcc=mcc)


def run_next_chunk():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: state=json.load(f)
    else:
        state={"results":[]}

    done_ids=set(r["config_id"] for r in state["results"])
    todo=[(i,band,cfg) for i,(band,cfg) in enumerate(ALL_CONFIGS) if i not in done_ids]
    total=len(ALL_CONFIGS)

    if not todo: print("All done."); return True

    t0=time.time()
    for i,(idx,band,cfg) in enumerate(todo[:CHUNK_SIZE]):
        p_in,p_out,hub,extra=cfg
        r=run_config(p_in,p_out,hub,extra,seeds=NEW_SEEDS)
        r["config_id"]=idx; r["band"]=band
        state["results"].append(r)
        done_now=len(state["results"]); remaining=total-done_now
        eta=(time.time()-t0)/(i+1)*remaining
        correct=(r["score"]>THRESHOLD)==(r["mean_delta"]>OUTCOME_MARGIN)
        print(f"  [{done_now:3d}/{total}] score={r['score']:.4f} band={band:4s} "
              f"Δ={r['mean_delta']:+.4f} {'SUBST' if r['mean_delta']>OUTCOME_MARGIN else 'NEGL ':5s} "
              f"pred={'✓' if correct else '✗'} ETA={eta:.0f}s")
        with open(CHECKPOINT,"w") as f: json.dump(state,f,indent=2)

    remaining=total-len(state["results"])
    if remaining>0:
        elapsed=time.time()-t0
        print(f"\nChunk done. {remaining} remain (~{elapsed/CHUNK_SIZE*remaining/60:.0f} min).")
        return False
    return True


def finalise():
    with open(CHECKPOINT) as f: state=json.load(f)
    results=state["results"]
    from scipy import stats as scipy_stats

    scores=np.array([r["score"] for r in results])
    deltas=np.array([r["mean_delta"] for r in results])
    r_val,p_val=scipy_stats.pearsonr(scores,deltas)
    m=cm_metrics(results)

    print("\n"+"="*68)
    print("FINAL PROSPECTIVE RULE VALIDATION")
    print(f"Rule: SCORE = homophily × deg_cv > {THRESHOLD}  (locked)")
    print(f"N={len(results)} new graphs | Seeds {NEW_SEEDS[:4]}...{NEW_SEEDS[-1]}")
    print("="*68)

    print(f"\nCorrelation: r={r_val:+.3f}  p={p_val:.2e}")
    print(f"\nConfusion Matrix:")
    print(f"  {'':22s}  Actual SUBST  Actual NEGL")
    print(f"  {'Predict SUBST':22s}  {m['TP']:>4} (TP)   {m['FP']:>4} (FP)")
    print(f"  {'Predict NEGL':22s}  {m['FN']:>4} (FN)   {m['TN']:>4} (TN)")
    print(f"\n  Accuracy:          {m['acc']:.1%}  ({m['TP']+m['TN']}/{m['n']})")
    print(f"  Balanced Accuracy: {m['bal']:.1%}")
    print(f"  MCC:               {m['mcc']:+.3f}")
    print(f"  Precision:         {m['prec']:.1%}")
    print(f"  Recall:            {m['rec']:.1%}")
    print(f"  Specificity:       {m['spec']:.1%}")

    # Calibration: predicted probability vs actual frequency
    # Bin by score quintile
    sorted_r=sorted(results,key=lambda x:x["score"])
    n=len(sorted_r); q=n//5
    print(f"\n  Calibration (by score quintile):")
    for qi in range(5):
        grp=sorted_r[qi*q:(qi+1)*q if qi<4 else n]
        mean_score=np.mean([g["score"] for g in grp])
        frac_subst=np.mean([g["mean_delta"]>OUTCOME_MARGIN for g in grp])
        frac_pred =np.mean([g["score"]>THRESHOLD for g in grp])
        print(f"    Q{qi+1}: mean_score={mean_score:.4f}  frac_subst={frac_subst:.1%}  frac_predicted_subst={frac_pred:.0%}")

    # FP/FN examples
    fps=[r for r in results if r["score"]>THRESHOLD and r["mean_delta"]<=OUTCOME_MARGIN]
    fns=[r for r in results if r["score"]<=THRESHOLD and r["mean_delta"]>OUTCOME_MARGIN]
    print(f"\n  False Positives ({len(fps)}) — predicted SUBST, actual NEGL:")
    for r in fps:
        print(f"    score={r['score']:.4f} H={r['homophily']:.3f} CV={r['deg_cv']:.3f} Δ={r['mean_delta']*100:+.2f}pp (just above threshold)")
    print(f"\n  False Negatives ({len(fns)}) — predicted NEGL, actual SUBST:")
    for r in fns:
        print(f"    score={r['score']:.4f} H={r['homophily']:.3f} CV={r['deg_cv']:.3f} Δ={r['mean_delta']*100:+.2f}pp (just below threshold)")

    # Combined with large_scale_v2
    with open("results/large_scale_v2.json") as f: v2=json.load(f)
    all_combined=results+v2["results"]
    m_comb=cm_metrics(all_combined)
    scores_c=np.array([r["score"] for r in all_combined])
    deltas_c=np.array([r["mean_delta"] for r in all_combined])
    r_c,p_c=scipy_stats.pearsonr(scores_c,deltas_c)
    print(f"\n  COMBINED (this + large_scale_v2 = {len(all_combined)} graphs):")
    print(f"    Accuracy={m_comb['acc']:.1%}  Bal.Acc={m_comb['bal']:.1%}  MCC={m_comb['mcc']:+.3f}")
    print(f"    r={r_c:+.3f}  p={p_c:.1e}")
    print(f"    TP={m_comb['TP']} FP={m_comb['FP']} FN={m_comb['FN']} TN={m_comb['TN']}")

    os.makedirs("results",exist_ok=True)
    out={
        "rule":"SCORE = homophily × deg_cv > 0.0505",
        "threshold":THRESHOLD,"outcome_margin_pp":OUTCOME_MARGIN*100,
        "seeds":NEW_SEEDS,"n":len(results),
        "correlation_r":float(r_val),"correlation_p":float(p_val),
        "metrics":m,"fp_examples":fps,"fn_examples":fns,
        "combined_with_v2":{"n":len(all_combined),"metrics":m_comb,"r":float(r_c),"p":float(p_c)},
        "results":results,
    }
    with open("results/rule_validation_final.json","w") as f: json.dump(out,f,indent=2)
    print("\nSaved → results/rule_validation_final.json")
    return out


if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser()
    p.add_argument("--finalise",action="store_true")
    args=p.parse_args()
    if args.finalise: finalise()
    else:
        done=run_next_chunk()
        if done: finalise()
