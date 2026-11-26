#!/usr/bin/env bash
# 전체 파이프라인. 저장소 루트에서 실행한다.
#   bash run_all.sh
#
# 데이터 경로는 configs/paths.yaml 과 각 스크립트의 인자로 지정한다.
# 단계별로 나눠 돌리려면 필요한 줄만 복사해 쓰면 된다.
set -euo pipefail

echo "═══ 1. 데이터 준비 ═══"
python scripts/make_val_split.py           # Tsinghua validation 을 dev/test 로 분할
python scripts/index_build.py              # Tsinghua 127클래스 인덱스
python scripts/download_uf_dogs.py         # UF 사진 받기 (재배포하지 않음)
python scripts/split_uf_panels.py          # 얼굴·전신 칸 분할
python scripts/parse_uf_dna.py             # DNA 표 파싱
python scripts/draft_uf_breed_map.py       # 품종 매핑 초안
python scripts/make_uf_eval_index.py       # 평가셋 인덱스
python scripts/build_uf_truth.py           # 정답 비율표
python scripts/build_top1_truth.py         # Top-1 정답표 (정확 매핑만)

echo "═══ 2. 특징 추출 ═══"
python -m src.features.extract_deep        # CLIP / DINOv2 / DINOv3
python scripts/sift_extract_train.py       # SIFT 코드북 + 학습·검증 특징
python scripts/import_sift_from_c.py       # SIFT 를 공용 특징 형식으로 변환
python scripts/make_uf_features.py         # UF 평가 입력 (SIFT·변형 이미지)

echo "═══ 3. 추정기 학습 ═══"
python scripts/train_from_feature_zips.py  # 분류기 (딥 특징)
python scripts/train_sift_npy.py           # 분류기 (SIFT)
python -m src.rq1.prototype_fcls           # 프로토타입 분해 (학습 없음)
python scripts/deep_mix_train.py           # 섞기 학습
python scripts/ablation_purebred.py        # 순종만 학습한 대조군

echo "═══ 4. 채점 (RQ1) ═══"
python scripts/evaluate_dev_top1.py
python scripts/evaluate_dev_ratio_metrics.py
python scripts/score_dev.py
python scripts/collect_rq1_preds.py        # 세 방법 예측 수집
python scripts/compare_rq1_dev.py          # RQ1 비교
python scripts/score_human_uf.py           # 사람 기준선
# 최종 1회 평가 — 설정을 모두 고정한 뒤에만 실행한다
# python scripts/evaluate_hold_once.py

echo "═══ 5. 단서 변형 (RQ2) ═══"
python scripts/prepare_d_inputs.py
python scripts/run_d_experiment.py
python scripts/validate_condition_matrix.py
python scripts/summarize_extra_drop.py
python scripts/apply_classifier_rq2.py     # 분류기를 변형 특징에 적용
python scripts/final_rq2.py                # RQ2 채점

echo "═══ 6. 표와 그림 ═══"
python scripts/summarize_classifier.py
python scripts/summarize_prototype.py
python scripts/report_figures.py
python scripts/plot_results.py

echo "완료"
