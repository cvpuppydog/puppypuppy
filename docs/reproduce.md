# 재현 절차

`run_all.sh`를 단계별로 풀어 쓴 문서다. 전체를 한 번에 돌리려면
`bash run_all.sh`만 해도 된다.

```bash
pip install -r requirements.txt
cp configs/paths.example.yaml configs/paths.yaml   # 데이터 경로를 적는다
```

Colab에서는 `notebooks/`의 노트북 3개가 아래 스크립트를 순서대로 부른다.

---

## 1. 데이터 준비

| 스크립트 | 하는 일 |
|---|---|
| `make_val_split.py` | Tsinghua validation 을 dev / test 로 나눈다 |
| `index_build.py` | Tsinghua 127클래스 인덱스를 만든다 (제외 3클래스: `chinese_rural_dog`, `dhole`, `African_hunting_dog`) |
| `download_uf_dogs.py` | UF 공개 페이지에서 사진표를 받는다 (재배포하지 않는다) |
| `split_uf_panels.py` | 개체당 한 장인 원본을 얼굴 칸·전신 칸으로 나눈다 |
| `parse_uf_dna.py` | DNA 결과 표를 파싱한다 |
| `draft_uf_breed_map.py` | 품종 매핑 초안을 만든다 → 손으로 확정해 `configs/breed_map.csv` |
| `make_uf_eval_index.py` | 평가셋 인덱스 |
| `build_uf_truth.py` | 정답 비율표 (DNA % / 100, 합이 100이 아니어도 늘리지 않는다) |
| `build_top1_truth.py` | Top-1 정답표 (`status=정확` 매핑만) |

학습 데이터는 Tsinghua Dogs, 평가 데이터는 UF 97마리(dev 20 / test 77)다.
UF 사진과 DNA는 저장소에 없다. Drive `00_COMMON` 또는 위 스크립트로 받는다.

## 2. 특징 추출

```bash
python -m src.features.extract_deep        # CLIP 512 / DINOv2 768 / DINOv3 768
python scripts/sift_extract_train.py       # SIFT BoVW 코드북(K=512) + 학습·검증 특징
python scripts/import_sift_from_c.py       # SIFT 를 공용 특징 형식으로 변환
python scripts/make_uf_features.py         # UF 평가 입력 (원본·변형 이미지)
```

특징은 5,000장 단위 묶음으로 원자적으로 저장되고 세 추정기가 공유한다.
전처리 규격은 `configs/preprocessing_features.md`(딥),
`configs/preprocessing_classifier.md`(분류기 입력)에 있다.

SIFT는 `bodybndbox` 로 잘라 336×336 흑백으로 바꾼 뒤 기술자를 뽑고,
L1 정규화 후 제곱근을 취한다(Hellinger / RootSIFT). 코드북은 학습 이미지
20,000장에서 MiniBatchKMeans 로 만든다. 재현을 위해 `random_state`를 고정하고
중심 할당은 최근접 중심으로 직접 계산한다 — 자세한 사정은
[`sift_analysis.md`](sift_analysis.md) 참고.

## 3. 추정기 학습

세 방법은 모두 같은 특징을 입력으로 받는 선형 모델이다.

| 방법 | 스크립트 | 학습 목표 |
|---|---|---|
| 순종 분류기 (A) | `train_from_feature_zips.py`, `train_sift_npy.py` | 127클래스 one-hot 교차엔트로피 |
| 프로토타입 분해 (B) | `python -m src.rq1.prototype_fcls` | 학습 없음. 클래스 평균 벡터에 FCLS 로 분해 |
| 합성 혼합 학습 (C) | `deep_mix_train.py` | 섞은 특징에 섞은 비율을 soft target 으로 |
| 대조군 (C) | `ablation_purebred.py` | 순종만, 같은 손실·같은 설정 |

합성 혼합: 품종 k∈{2,3,4}개를 뽑아 Dirichlet(α=1.0) 가중치 w로
`x_mix = Σ wⱼ·xⱼ`, 목표는 w 자체. 에폭마다 혼합 20,000개 + 순종 전체를 쓴다.
`nn.Linear(d,127)`, AdamW lr 3e-3, wd 1e-4, 20 에폭, batch 512, 시드 1·2·3.
대조군은 혼합을 뺀 것만 다르다(혼합 ≈3,200 step, 대조군 ≈2,420 step).

학습을 건너뛰려면 `models/`의 가중치를 그대로 쓴다.

> `torch.randperm`의 난수는 장치에 따라 다르다. GPU에서 학습한 체크포인트는
> GPU에서만 똑같이 재현된다(T4에서 확률 최대 차이 0.000000 확인). CPU에서
> 다시 돌리면 가중치가 달라진다.

## 4. 채점 (RQ1)

```bash
python scripts/evaluate_dev_top1.py
python scripts/evaluate_dev_ratio_metrics.py
python scripts/score_dev.py
python scripts/collect_rq1_preds.py        # 세 방법의 개체 단위 예측을 한 형식으로
python scripts/compare_rq1_dev.py          # 같은 코드로 한 번에 채점
python scripts/score_human_uf.py           # 사람 기준선
```

지표 정의는 [`classifier_metrics.md`](classifier_metrics.md)가 기준이다.

- **주 지표** — 원래 DNA 비율 기준 포착 혈통 비율@3 + coverage
- 보조 — 재정규화 포착@3, 재정규화 분포 겹침 `Σ min(pred, truth_norm)`,
  Top-1 + Wilson 95% CI
- 기준선 — 찍기(학습 클래스 빈도), 상한(정확 매핑된 DNA 비율 중 큰 3개의 합)
- 판정 — 개체 단위 짝 차이 평균 ± 2SE. 구간이 0을 포함하면 판정 보류
- 매핑은 `status=정확`만 쓴다. 푸들 계열 4클래스는 top-3 전에 합친다
  (Top-1은 합치기 전 127클래스로 낸다)

시드 3개의 개체별 확률을 평균한 뒤 채점한 값이 대표값이고, 시드별 최소~최대
범위를 함께 본다.

**`evaluate_hold_once.py`는 설정을 전부 고정한 뒤 한 번만 돌린다.**
기록은 `results/classifier/HOLD_EVALUATED_ONCE.json`,
규칙은 [`classifier_hold_lock.md`](classifier_hold_lock.md).

## 5. 단서 변형 (RQ2)

```bash
python scripts/prepare_d_inputs.py
python scripts/run_d_experiment.py
python scripts/validate_condition_matrix.py
python scripts/summarize_extra_drop.py
python scripts/apply_classifier_rq2.py     # 분류기를 변형 특징에 적용
python scripts/final_rq2.py
```

조건은 `configs/conditions.yaml`에 있다. 원본, 얼굴 칸, 전신 칸, 배경,
블러(Gaussian σ = 1, 3, 6, 12), 흑백. 변형 이미지는 PNG로 저장해 재저장
손실이 생기지 않게 한다. 흑백은 SIFT에서 원본과 동일해지므로
SIFT 조건에서 뺀다.

> 얼굴·전신은 UF 원본의 좌우 칸을 나눈 것이고 분할 마스크가 아니다.
> 문서와 결과에서 `panel_face` / `panel_body`로 적으며 영역 제거 실험으로
> 해석하지 않는다.

## 6. 표와 그림

```bash
python scripts/summarize_classifier.py
python scripts/summarize_prototype.py
python scripts/report_figures.py
python scripts/plot_results.py
```

결과는 `results/`, 그림은 `figures/`에 들어간다.
원고는 `report/cvpr/build_cvpr_docx.py`로 빌드한다.

## 7. 테스트

```bash
python -m pytest tests/ -q
```
