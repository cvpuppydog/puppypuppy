"""
dev 채점 (팀 평가 지표 정의 기준). hold 는 사용하지 않는다.

지표
  주 지표   : 원래 DNA 비율 포착@3 (재정규화 없음), coverage 병기
  보조      : 재정규화 포착@3, 재정규화 분포 겹침, Top-1 (Wilson 95%)
  기준선    : 찍기(학습 데이터 클래스 빈도), 상한(정확 매핑 DNA 상위 3개)
  판단      : 찍기 대비 개체 단위 짝 차이 ± 2SE 가 0을 포함하면 판단 불가

입력 (<WORK> 또는 현재 폴더)
  pred_eval_uf-{ext}-mix-s{1,2,3}.csv   모델 예측 (개체 단위)
  dna_UF.csv                            전체 DNA 혈통 비율 (정답)
  breed_map.csv                         DNA 품종명 → Tsinghua 클래스, status=정확만 사용
  index_local_v2.csv                    찍기 기준선용 클래스 빈도

출력
  <WORK>/RQ1_dev_metrics.csv
  <WORK>/perdog_dev_{ext}.csv

실행
  python score_dev.py
"""
import os
import glob
import numpy as np
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import WORK

POODLE = {"teddy", "toy_poodle", "miniature_poodle", "standard_poodle"}
EXTS = ["sift", "clip", "dinov2", "dinov3"]


def find(name):
    """<WORK> 또는 현재 폴더에서 파일을 찾는다."""
    for p in (f"{WORK}/{name}", name, f"/content/{name}"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


# ── 매핑: status=정확만 사용 ──────────────────────────────────────
bm = pd.read_csv(find("breed_map.csv"))
bm = bm[bm["status"] == "정확"]
BMAP = {r.breed_raw: [t.strip() for t in str(r.tsinghua_classes).split(";")]
        for r in bm.itertuples() if isinstance(r.tsinghua_classes, str)}


def dkey(breed):
    """DNA 품종명을 채점용 키로. Poodle 계열은 하나로 합친다."""
    if breed not in BMAP:
        return None
    cs = BMAP[breed]
    return "Poodle" if any(c in POODLE for c in cs) else cs[0]


# ── 정답 ──────────────────────────────────────────────────────────
dna = pd.read_csv(find("dna_UF.csv"))
dna["pct"] = pd.to_numeric(dna["pct"], errors="coerce").fillna(0.0)
dna["key"] = dna["breed_raw"].map(dkey)
dna["p"] = dna["pct"] / 100
T = dna.dropna(subset=["key"]).groupby(["dog_id", "key"])["p"].sum().unstack(fill_value=0.0)

# Top-1 정답: 전체 DNA 에서 공동 1위를 찾은 뒤 정확 매핑 클래스 합집합
TOP1 = {}
for d, g in dna.groupby("dog_id"):
    m = g["pct"].max()
    cls = set()
    for b in g.loc[g["pct"] == m, "breed_raw"]:
        if b in BMAP:
            cls |= set(BMAP[b])
    TOP1[d] = cls if cls else None


def merge(p):
    """같은 DNA 품종으로 묶인 Tsinghua 클래스의 확률을 먼저 합산한다."""
    out = {}
    for c, v in p.items():
        k = "Poodle" if c in POODLE else c
        out[k] = out.get(k, 0.0) + v
    return pd.Series(out)


def score_table(P):
    """개체별 지표. P: index=dog_id, columns=127 클래스."""
    recs = []
    for d, p in P.iterrows():
        pm = merge(p)
        tr = T.loc[d] if d in T.index else pd.Series(dtype=float)
        tsum = float(tr.sum()) if len(tr) else 0.0
        top3 = pm.nlargest(3).index
        cap = float(tr.reindex(top3).fillna(0).sum()) if len(tr) else 0.0
        if tsum > 0:
            tn = tr / tsum
            capn = float(tn.reindex(top3).fillna(0).sum())
            keys = pm.index.union(tn.index)
            ov = float(np.minimum(pm.reindex(keys).fillna(0), tn.reindex(keys).fillna(0)).sum())
        else:
            capn, ov = np.nan, np.nan
        ts = TOP1.get(d)
        recs.append({"dog_id": d, "cap3": cap, "tsum": tsum, "capn3": capn,
                     "overlap": ov, "top1": np.nan if not ts else float(p.idxmax() in ts)})
    return pd.DataFrame(recs).set_index("dog_id")


def wilson(k, n):
    if not n:
        return (np.nan, np.nan)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 3), round(c + h, 3))


def summarize(df):
    ev = df["top1"].dropna()
    n, k = len(ev), int(ev.sum())
    return {"captured@3": round(df["cap3"].mean(), 4),
            "coverage_DNA": round(df["tsum"].mean(), 4),
            "captured_norm@3": round(df["capn3"].mean(), 4),
            "overlap": round(df["overlap"].mean(), 4),
            "top1": f"{k}/{n}",
            "top1_acc": round(k / n, 4) if n else np.nan,
            "top1_CI": wilson(k, n),
            "coverage_scorable": f"{n}/{len(df)}"}


def find_pred(ext, seed):
    """UF 예측 파일을 찾는다.

    코드가 새로 생성한 이름(pred_eval_uf-{ext}-mix-s{n}.csv)과
    저장소에 동봉된 이름(pred_eval_uf-{ext}-mix-v2-s{n}.csv,
    pred_eval_uf_v2-sift-mix-s{n}.csv)을 모두 받아들인다.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = [WORK, os.path.join(here, "..", "results", "pred"),
            os.path.join(here, "results", "pred"), "."]
    pats = [f"pred_eval_uf-{ext}-mix-s{seed}.csv",
            f"pred_eval_uf-{ext}-mix-v2-s{seed}.csv",
            f"pred_eval_uf*{ext}-mix*s{seed}.csv"]
    for d in dirs:
        for p in pats:
            hit = sorted(glob.glob(os.path.join(d, p)))
            if hit:
                return hit[0]
    raise FileNotFoundError(f"{ext} 시드 {seed} 예측 파일을 찾을 수 없다")


def load(ext, seed):
    x = pd.read_csv(find_pred(ext, seed))
    return x[x["split"] == "dev"].set_index("dog_id").drop(columns=["split"])


def main():
    rows, cols, dev_ids = [], None, None
    for ext in EXTS:
        try:
            Ps = [load(ext, s) for s in (1, 2, 3)]
        except FileNotFoundError:
            print("건너뜀 (예측 없음):", ext)
            continue
        cols = Ps[0].columns
        Pm = sum(P[cols] for P in Ps) / 3
        sc = score_table(Pm)
        sc.to_csv(f"{WORK}/perdog_dev_{ext}.csv")
        seed_sc = [summarize(score_table(P[cols])) for P in Ps]
        rows.append({"방법": ext, **summarize(sc),
                     "seed범위": (min(s["captured@3"] for s in seed_sc),
                                  max(s["captured@3"] for s in seed_sc))})
        dev_ids = Pm.index

    if not rows:
        print("예측 파일이 없다. deep_mix_train.py 를 먼저 실행할 것.")
        return

    # 찍기 기준선: 학습 데이터 클래스 빈도 (인덱스가 있을 때만)
    ch = None
    try:
        idx = pd.read_csv(find("index_local_v2.csv"))
        freq = (idx[(idx["split"] == "train") & idx["keep_127"]]["breed_name"]
                .value_counts(normalize=True).reindex(cols).fillna(0))
        Pch = pd.DataFrame([freq.values] * len(dev_ids), index=dev_ids, columns=cols)
        ch = score_table(Pch)
        rows.append({"방법": "찍기", **summarize(ch)})
    except FileNotFoundError:
        print("[안내] index_local_v2.csv 가 없어 찍기 기준선을 건너뛴다.")
        print("       보고된 찍기 값은 results/metrics/RQ1_dev_metrics_final.csv 에 있다.")

    # 상한: 정확 매핑 DNA 상위 3개
    ub = np.mean([T.loc[d].nlargest(3).sum() for d in dev_ids if d in T.index])
    rows.append({"방법": "상한", "captured@3": round(ub, 4)})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    print(res.to_string(index=False))
    res.to_csv(f"{WORK}/RQ1_dev_metrics.csv", index=False)

    # 찍기 대비 판단
    if ch is None:
        return
    print("\n■ 찍기 대비 (개체 단위 짝 차이 ± 2SE)")
    cmp = []
    for ext in EXTS:
        f = f"{WORK}/perdog_dev_{ext}.csv"
        if not os.path.exists(f):
            continue
        m = pd.read_csv(f).set_index("dog_id")
        for col in ["cap3", "capn3", "overlap"]:
            d = (m[col] - ch.loc[m.index, col]).dropna()
            mean = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
            lo, hi = mean - 2 * se, mean + 2 * se
            cmp.append({"방법": ext, "지표": col, "차이": round(mean, 4),
                        "±2SE": round(2 * se, 4),
                        "판단": "판단 불가" if lo <= 0 <= hi else
                                ("찍기보다 높음" if lo > 0 else "찍기보다 낮음")})
    print(pd.DataFrame(cmp).to_string(index=False))


if __name__ == "__main__":
    main()
