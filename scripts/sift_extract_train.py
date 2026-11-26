"""
SIFT + BoVW 특징 추출 (코드북 생성 + 학습/검증 전체 추출).

출력
  <WORK>/sift_codebook.npy   (K x 128) 코드북
  <WORK>/sift_all_v3.npy     (N x K)   전체 사진의 BoVW 특징, index_local_v2.csv 행 순서와 일치
  <WORK>/sift_v3_chunks/     중간 저장 (중단 후 재실행하면 이어서 진행)

절차
  1. 어노테이션의 몸 박스로 크롭 → CROP_SIZE x CROP_SIZE 회색조
  2. SIFT 기술자 검출
  3. 학습 사진 CODEBOOK_SAMPLE 장의 기술자로 MiniBatchKMeans 코드북 학습 (K = CODEBOOK_K)
  4. 각 기술자를 가장 가까운 코드워드에 배정 → 히스토그램
  5. L1 정규화 후 제곱근 변환 (Hellinger / RootSIFT 계열)

[중요] 보고된 결과와의 차이
  본 저장소의 보고 수치는 이전 버전(sift_all_v2.npy)으로 산출되었다.
  그 버전은 저장된 코드북을 그대로 쓰지 않고
      MiniBatchKMeans(init=centers, n_init=1, max_iter=1).fit(centers)
  로 한 번 더 학습한 뒤 predict 했는데, random_state 가 없어 단어 공간이
  실행마다 달라지고 재현되지 않았다. 본 스크립트는 저장된 코드북에
  직접 최근접 배정하여 이 문제를 제거한 버전이다.
  상세는 docs/sift_analysis.md 참조.

실행
  python sift_extract_train.py
"""
import os
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool
from sklearn.cluster import MiniBatchKMeans

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import WORK, CODEBOOK_K, CODEBOOK_SAMPLE, CROP_SIZE

CHUNK = 5000
CHUNK_DIR = f"{WORK}/sift_v3_chunks"
CB_PATH = f"{WORK}/sift_codebook.npy"
OUT_PATH = f"{WORK}/sift_all_v3.npy"


def descriptors(path, x1, y1, x2, y2, sift):
    """몸 박스로 크롭한 회색조 이미지에서 SIFT 기술자를 뽑는다."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return None
    _, des = sift.detectAndCompute(cv2.resize(crop, (CROP_SIZE, CROP_SIZE)), None)
    return des


def build_codebook(idx):
    """학습 사진에서 기술자를 모아 코드북을 만든다. 이미 있으면 재사용한다."""
    if os.path.exists(CB_PATH):
        print("코드북 재사용:", CB_PATH)
        return np.load(CB_PATH).astype(np.float32)

    sift = cv2.SIFT_create()
    sub = idx[idx["split"] == "train"].sample(CODEBOOK_SAMPLE, random_state=0)
    pool = []
    for r in sub.itertuples():
        d = descriptors(r.path, r.x1, r.y1, r.x2, r.y2, sift)
        if d is not None:
            pool.append(d)
    km = MiniBatchKMeans(n_clusters=CODEBOOK_K, batch_size=4096,
                         random_state=0, n_init=3).fit(np.vstack(pool))
    np.save(CB_PATH, km.cluster_centers_)
    print("코드북 생성:", CB_PATH, km.cluster_centers_.shape)
    return km.cluster_centers_.astype(np.float32)


# 워커 전역 (Pool initializer 로 설정)
_SIFT = None
_CB = None
_CBSQ = None


def _init(cb):
    global _SIFT, _CB, _CBSQ
    _SIFT = cv2.SIFT_create()
    _CB = cb
    _CBSQ = (cb ** 2).sum(1)


def _feat(args):
    """한 장의 BoVW 히스토그램. 저장된 코드북에 직접 최근접 배정한다."""
    path, x1, y1, x2, y2 = args
    h = np.zeros(len(_CB), np.float32)
    d = descriptors(path, x1, y1, x2, y2, _SIFT)
    if d is not None and len(d):
        d = d.astype(np.float32)
        lab = ((d ** 2).sum(1, keepdims=True) - 2 * d @ _CB.T + _CBSQ).argmin(1)
        h = np.bincount(lab, minlength=len(_CB)).astype(np.float32)
        h /= max(h.sum(), 1)
        h = np.sqrt(h)
    return h


def main():
    idx = pd.read_csv(f"{WORK}/index_local_v2.csv")
    cb = build_codebook(idx)

    os.makedirs(CHUNK_DIR, exist_ok=True)
    rows = list(zip(idx["path"], idx["x1"], idx["y1"], idx["x2"], idx["y2"]))
    n_chunks = (len(rows) + CHUNK - 1) // CHUNK

    with Pool(os.cpu_count(), initializer=_init, initargs=(cb,)) as pool:
        for k in range(0, len(rows), CHUNK):
            f = f"{CHUNK_DIR}/chunk_{k // CHUNK:03d}.npy"
            if os.path.exists(f):
                continue
            np.save(f, np.stack(pool.map(_feat, rows[k:k + CHUNK], chunksize=64)))
            print(f"chunk {k // CHUNK:02d} 완료 ({min(k + CHUNK, len(rows))}/{len(rows)})", flush=True)

    X = np.concatenate([np.load(f"{CHUNK_DIR}/chunk_{i:03d}.npy") for i in range(n_chunks)])
    np.save(OUT_PATH, X)
    print("완료", X.shape, "| 한 번도 안 쓰인 시각 단어:", int((X.max(0) == 0).sum()), "/", CODEBOOK_K)


if __name__ == "__main__":
    main()
