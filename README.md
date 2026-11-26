# 사진으로 믹스견의 품종 혼합 비율 추정

컴퓨터비전 팀 프로젝트. 사진 한 장에서 개의 품종 혼합 비율을 추정한다.

- **RQ1** — 합성 혼합으로 학습한 추정기가 순종 분류기나 고정 프로토타입 분해보다 나은가
- **RQ2** — 어떤 시각적 단서(얼굴·몸·배경, 블러, 흑백)가 추정에 기여하는가

원고: [`report/cvpr/paper_en.pdf`](report/cvpr/paper_en.pdf) · [`report/cvpr/paper_ko.pdf`](report/cvpr/paper_ko.pdf)

---

## 1. 실행 환경

Python 3.10 이상. Colab과 로컬(VS Code) 모두에서 돌아간다.

```bash
pip install -r requirements.txt
```

딥 특징 추출에는 GPU가 있으면 빠르지만 없어도 돌아간다. 공개 체크포인트
(`openai/clip-vit-base-patch16`, `facebook/dinov2-base`,
`facebook/dinov3-vitb16-pretrain-lvd1689m`)를 내려받으므로 처음 실행할 때
네트워크가 필요하다.

데이터 경로는 [`configs/paths.yaml`](configs/paths.yaml)에서 지정한다.
`configs/paths.example.yaml`을 복사해 쓰면 된다.

## 2. 폴더

| 폴더 | 내용 |
|---|---|
| `configs/` | 품종 매핑, 127클래스 목록, 백본·조건 설정, 전처리 규격 |
| `src/` | 공용 라이브러리 (`common` 설정·입출력, `features` 추출, `rq1` 프로토타입 분해, `demo`) |
| `scripts/` | 실행 스크립트. 파이프라인 단계별로 `run_all.sh` 가 순서대로 부른다 |
| `models/` | 학습된 선형 가중치. `classifier/` 12개(`.npz`), `mixing/` 12개(`.pt`) |
| `results/` | 집계 결과. `classifier/ prototype/ mixing/ rq2/ logs/` |
| `figures/` | 표와 그림 |
| `docs/` | 방법론, 평가 정의, 결과, 분석 |
| `report/` | 원고 소스(`cvpr/` 영문·국문, `ko/` 긴 국문 보고서)와 그림 |
| `notebooks/` | Colab 노트북 3개. 스크립트를 부르기만 한다 |
| `tests/` | 단위 테스트 (`python -m pytest tests/`) |
| `demo/` | Gradio 데모 빌드 |

## 3. 실행 순서

전체는 [`run_all.sh`](run_all.sh)에 단계별로 적혀 있다.

```bash
bash run_all.sh            # 1 데이터 → 2 특징 → 3 학습 → 4 채점 → 5 RQ2 → 6 표·그림
```

단계별 설명은 [`docs/reproduce.md`](docs/reproduce.md)에 있다.
학습을 건너뛰고 채점만 해 보려면 `models/`의 가중치를 그대로 쓸 수 있다.

**최종 1회 평가**(`scripts/evaluate_hold_once.py`)는 `run_all.sh`에서 주석으로
막아 두었다. 모든 설정을 고정한 뒤에만 한 번 돌린다
([`docs/classifier_hold_lock.md`](docs/classifier_hold_lock.md)).

## 4. 학습된 모델

| 폴더 | 파일 | 형식 | 설명 |
|---|---|---|---|
| `models/classifier/` | `RQ1-{clip,dinov2,dinov3,sift}-clf-s{1,2,3}_linear_weights.npz` | `weight (127,d)`, `bias (127,)`, `classes (127,)` | 순종 분류기. 입력 전처리는 행별 L2 정규화 |
| `models/mixing/` | `RQ1-{clip,dinov2,dinov3,sift}-mix-v2-s{1,2,3}-lr3e-3.pt` | `state`, `classes`, `mu`, `sd` | 합성 혼합 학습 선형 헤드. 표준화 통계가 체크포인트에 들어 있다 |

각 폴더의 `README`에 입력 규격과 추론 식이 있다. 합계 7.4MB라 저장소에 둔다.

## 5. 데이터 정책

UF(플로리다 대학교 보호소 의학 프로그램) 자료의 이용 조건에 따라 **개별 사진,
개체별 DNA 결과, 그로부터 만든 특징을 이 저장소에 올리지 않는다.** 집계 결과만
보고한다. 아래는 저장소에 없고 팀 Drive에 있다.

| 없는 것 | 어디에 |
|---|---|
| UF 사진, DNA 정답(`dna_UF.csv`), `split.csv`, 개체별 index | Drive `00_COMMON` |
| 특징 `.npy` 묶음, SIFT 코드북 | Drive `B`, `C` |
| 개 ID가 붙은 개체별 예측·점수 | Drive 각 파트 |
| SAM3 체크포인트, 마스크, 변형 이미지 | Drive `D` |
| 발표자료, 시연 영상 | Drive `99_REPORT_PRESENTATION` |

`scripts/download_uf_dogs.py`가 공개 페이지에서 직접 받아 온다.
자료 제공처에 사용 허가를 요청했고 회신을 기다리고 있다.

## 6. 담당

| 담당 | 맡은 부분 | 문서 |
|---|---|---|
| A | 순종 분류기, 평가 정의와 채점 | `docs/classifier_*.md`, `docs/evaluation_policy.md` |
| B | 특징 추출, 프로토타입 분해(FCLS), 원고 | `docs/prototype_*.md`, `report/` |
| C | 합성 혼합 학습, SIFT BoVW, SIFT 실패 분석 | `docs/mixing_*.md`, `docs/sift_analysis.md` |
| D | RQ2 단서 변형 실험 | `docs/rq2_*.md` |

## 7. 라이선스

코드는 MIT ([`LICENSE`](LICENSE)). UF 자료와 Tsinghua Dogs 데이터셋은 각자의
이용 조건을 따르며 이 라이선스가 적용되지 않는다.
