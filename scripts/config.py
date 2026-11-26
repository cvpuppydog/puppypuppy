"""
경로와 공통 설정. 실행 환경에 맞춰 이 파일만 고치면 된다.
"""
import os

# ── 작업 폴더 (모든 산출물이 여기에 저장된다) ──────────────────────────
WORK = os.environ.get("C_WORK", "/content/drive/MyDrive/C_work")

# ── 입력 데이터 ────────────────────────────────────────────────────
# Tsinghua Dogs 이미지 루트: <IMAGE_ROOT>/<품종 폴더>/*.jpg
IMAGE_ROOT = os.environ.get("C_IMAGES", "/content/low-resolution")
# Tsinghua Dogs 어노테이션 루트 (XML)
ANN_ROOT = os.environ.get("C_ANN", "/content/ann")
# train.lst / validation.lst 가 있는 폴더
LST_DIR = os.environ.get("C_LST", "/content/ann/TrainAndValList")
# UF 평가셋 루트: <EVALSET>/raw/UF/panels/*.jpg, <EVALSET>/eval_index_UF_panels.csv
EVALSET = os.environ.get("C_EVALSET", "/content/evalset")

# ── 학습 설정 (논문 보고 수치와 동일) ───────────────────────────────
N_CLASSES = 127
EXCLUDE = {"chinese_rural_dog", "dhole", "African_hunting_dog"}

CODEBOOK_K = 512          # SIFT BoVW 코드북 크기
CODEBOOK_SAMPLE = 20000   # 코드북 학습에 쓰는 학습 사진 수
CROP_SIZE = 336           # 크롭 후 리사이즈 한 변

ALPHA = 1.0               # Dirichlet 농도
KMAX = 4                  # 혼합 품종 수 상한 (k in [2, KMAX])
LR = 3e-3
EPOCHS = 20
N_MIX = 20000             # 에폭당 합성 믹스 수
BATCH = 512
WEIGHT_DECAY = 1e-4
SEEDS = (1, 2, 3)

BLUR_SIGMAS = [1, 3, 6, 12]   # RQ2용 블러 단계
