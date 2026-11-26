"""
혼합 학습 (RQ1). 고정 특징 위의 선형 헤드를 합성 믹스로 학습한다.

사용법
  python deep_mix_train.py clip        # clip / dinov2 / dinov3 / sift
  python deep_mix_train.py             # 네 추출기를 차례로 실행

출력 (<WORK> 아래)
  RQ1-{ext}-mix-s{seed}.pt              체크포인트
  pred_RQ1-val-{ext}-mix-s{seed}.csv    순종 validation dev 예측 (2,540행)
  pred_eval_uf-{ext}-mix-s{seed}.csv    UF 개체 단위 예측 (100마리)

[재현 참고]
  본 저장소의 보고 수치는 GPU에서 산출되었다. torch.randperm 의 난수열이
  장치(CPU/GPU)와 torch 버전에 따라 달라지므로, 다른 환경에서 재실행하면
  가중치가 미세하게 달라질 수 있다. 확인된 예: CLIP 시드 1을 CPU에서
  재학습했을 때 1위 예측 일치율 88.2%, dev top-1 0.7516 -> 0.7598.
  보고 수치의 근거가 되는 예측 CSV 는 results/pred/ 에 보존되어 있다.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import (WORK, ALPHA, KMAX, LR, EPOCHS, N_MIX, BATCH,
                    WEIGHT_DECAY, SEEDS)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_index():
    df = pd.read_csv(f"{WORK}/index_local_v2.csv")
    df["image_id"] = (df["path"].str.extract(r"(n\d{6})")[0] + "/"
                      + df["path"].str.extract(r"/(n\d+)\.[a-z]+$")[0])
    return df


def load_features(ext, df):
    """추출기별 특징 행렬을 반환한다. 행 순서는 df 와 같다."""
    if ext == "sift":
        X = np.load(f"{WORK}/sift_all_v3.npy").astype(np.float32)
        assert len(X) == len(df), "특징 행 수와 인덱스 행 수가 다르다"
        uf_idx = pd.read_csv(f"{WORK}/index_eval_uf_v5_all.csv").query("condition == 'orig'").reset_index(drop=True)
        U = np.load(f"{WORK}/sift_eval_uf_v5_all.npy")[:len(uf_idx)].astype(np.float32)
        return (lambda m: X[m.values if hasattr(m, "values") else m]), U, uf_idx

    # 딥 특징: B 가 제공한 샤드 (image_id + cls)
    fe = f"/content/{ext}_feat/{ext}"
    if not os.path.exists(fe):
        os.system(f'unzip -q "{WORK}/{ext}.zip" -d /content/{ext}_feat')
    emb = {}
    for sp in ["train", "val", "eval_uf_panels"]:
        for s in sorted(glob.glob(f"{fe}/{sp}/shard_*.npz")):
            z = np.load(s)
            emb.update(zip(z["image_id"], z["cls"].astype(np.float32)))
    uf_idx = pd.read_csv(f"{WORK}/index_eval_uf.csv")
    U = np.stack([emb[i] for i in uf_idx["image_id"]])
    return (lambda m: np.stack([emb[i] for i in df.loc[m, "image_id"]])), U, uf_idx


def run(ext):
    df = load_index()
    classes = sorted(df.loc[df["keep_127"], "breed_name"].unique())
    cid = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)

    tr = (df["keep_127"] & (df["split"] == "train")).values
    va = (df["keep_127"] & (df["split"] == "validation")).values
    dev_ids = set(pd.read_csv(f"{WORK}/val_split_seed1.csv").query("split == 'dev'")["image_id"])
    dsel = va & df["image_id"].isin(dev_ids).values

    get, U, uf_idx = load_features(ext, df)
    X_tr = get(tr)
    y_tr = df.loc[tr, "breed_name"].map(cid).values
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (get(va) - mu) / sd
    y_va = df.loc[va, "breed_name"].map(cid).values
    X_dev = (get(dsel) - mu) / sd
    meta_dev = df.loc[dsel, ["image_id", "breed_name"]].reset_index(drop=True)
    X_uf = (U - mu) / sd
    by_cls = [np.where(y_tr == c)[0] for c in range(n_cls)]
    print(f"{ext} | 장치 {DEV} | 클래스 {n_cls} | train {X_tr.shape} | dev {X_dev.shape} | UF {X_uf.shape}")

    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        def make_mixed(n):
            """품종 k개(k in [2, KMAX])를 Dirichlet 비율로 가중합한 합성 믹스."""
            fs, ts = [], []
            for _ in range(n):
                k = rng.integers(2, KMAX + 1)
                cl = rng.choice(n_cls, k, replace=False)
                r = rng.dirichlet([ALPHA] * k)
                f = sum(w * X_tr[rng.choice(by_cls[c])] for c, w in zip(cl, r))
                t = np.zeros(n_cls, np.float32)
                t[cl] = r
                fs.append(f); ts.append(t)
            return np.stack(fs).astype(np.float32), np.stack(ts)

        model = nn.Linear(X_tr.shape[1], n_cls).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        onehot = np.eye(n_cls, dtype=np.float32)[y_tr]

        for _ in range(EPOCHS):
            Xm, Tm = make_mixed(N_MIX)
            Xb = torch.tensor(np.vstack([X_tr, Xm]), device=DEV)
            Tb = torch.tensor(np.vstack([onehot, Tm]), device=DEV)
            perm = torch.randperm(len(Xb), device=DEV)
            for i in range(0, len(Xb), BATCH):
                b = perm[i:i + BATCH]
                loss = -(Tb[b] * F.log_softmax(model(Xb[b]), -1)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
            P_va = F.softmax(model(t(X_va)), -1).cpu().numpy()
            P_dev = F.softmax(model(t(X_dev)), -1).cpu().numpy()
            P_uf = F.softmax(model(t(X_uf)), -1).cpu().numpy()
        print(f"  seed {seed} | val top-1 {(P_va.argmax(1) == y_va).mean():.4f}")

        # 체크포인트 (추론에 필요한 모든 것)
        torch.save({"state": model.state_dict(), "classes": classes,
                    "mu": mu, "sd": sd, "seed": seed, "lr": LR,
                    "epochs": EPOCHS, "kmax": KMAX, "alpha": ALPHA,
                    "extractor": ext},
                   f"{WORK}/RQ1-{ext}-mix-s{seed}.pt")

        # 순종 dev 예측
        out = pd.DataFrame(P_dev, columns=classes)
        out.insert(0, "split", "dev")
        out.insert(0, "label", meta_dev["breed_name"].values)
        out.insert(0, "image_id", meta_dev["image_id"].values)
        out.to_csv(f"{WORK}/pred_RQ1-val-{ext}-mix-s{seed}.csv", index=False)

        # UF 개체 단위 예측 (얼굴·전신 평균 후 재정규화)
        e = pd.DataFrame(P_uf, columns=classes)
        e.insert(0, "split", uf_idx["split"].values)
        e.insert(0, "dog_id", uf_idx["dog_id"].values)
        dog = e.groupby(["dog_id", "split"], as_index=False).mean(numeric_only=True)
        dog[classes] = dog[classes].div(dog[classes].sum(1), axis=0)
        dog.to_csv(f"{WORK}/pred_eval_uf-{ext}-mix-s{seed}.csv", index=False)
        print(f"    저장 완료 | dev {len(out)}행 | UF {len(dog)}마리")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["sift", "clip", "dinov2", "dinov3"]
    for e in targets:
        run(e)
