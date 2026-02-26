# CALF Soccer Highlight Pipeline (Qwen VLM + TTS + Search)

## 1. 프로젝트 개요

본 프로젝트는 축구 중계 영상을 입력받아 이벤트 기반 하이라이트를 자동 생성하고, 장면 검색 및 음성 내레이션까지 포함하는 멀티모달 파이프라인이다.

핵심 목표는 다음과 같다.

- 경기 전체 영상에서 유의미한 이벤트를 자동 검출
- 이벤트 전후 구간을 클립으로 추출하고 최종 하이라이트 영상 구성
- 장면 텍스트 설명 및 프롬프트 기반 검색 지원
- TTS 생성 결과를 하이라이트 오디오에 반영
- 비개발자도 사용할 수 있는 단계형 웹 UI 제공

---

## 2. 시스템 아키텍처

파이프라인은 크게 6개 계층으로 구성된다.

1. 이벤트 검출 계층 (CALF)
2. 클립 추출/병합 계층 (FFmpeg)
3. 장면 설명 계층 (Qwen VLM)
4. 검색 인덱싱/검색 계층 (BM25 / Semantic / Hybrid)
5. TTS 계층 (Qwen3-TTS 기반)
6. 웹 애플리케이션 계층 (Gradio)

주요 진입점:

- 웹 UI: `inference/app.py`
- 파이프라인 오케스트레이션: `inference/enhanced_highlight_generator.py`

---

## 3. 기술 상세

### 3.1 이벤트 검출 (CALF)

CALF 모델은 축구 이벤트를 프레임 시점으로 탐지한다.

- 지원 이벤트: 17종 (Goal, Foul, Corner 등)
- 출력: `timestamp`, `event_type`, `confidence`
- 후처리: NMS 적용 후 임계치 기반 필터링

추가 정책:

- 동일 `event_type` 간 시간 구간이 겹치면 신뢰도 높은 이벤트만 유지
- `Full Highlight Video` 생성 시에는 이벤트 타입과 무관하게 시간 구간 중복이 없도록 대표 클립만 선택

### 3.2 클립 추출 및 하이라이트 병합

탐지된 이벤트를 기준으로 전후 구간을 클립으로 추출한다.

- 기본 윈도우: `before_event`, `after_event`
- 클립 추출: FFmpeg
- 최종 병합: concat 기반 병합, 필요 시 재인코딩 fallback

출력 파일 예시:

- `clips/clip_XXX_*.mp4`
- `highlight.mp4`
- TTS 적용 시 `highlight_tts.mp4`

### 3.3 장면 설명 (Qwen VLM)

VLM은 클립 프레임 샘플을 받아 장면 설명 텍스트를 생성한다.

- 프레임 샘플링 후 텍스트 프롬프트 결합
- 한국어/영어 라우팅 지원 (`auto`, `ko`, `en`)
- 로컬 모델 경로 또는 HF 모델 ID 모두 지원

운영상 고려사항:

- 대형 로컬 VLM은 VRAM 제약으로 로딩 지연/실패 가능
- CPU offload 모드 지원 (Windows에서는 매우 느릴 수 있음)
- 설명 실패 시 파이프라인 중단 대신 fallback 설명 생성으로 다음 단계 지속

### 3.4 검색 엔진

검색은 `event_type + description` 텍스트를 기반으로 인덱싱한다.

지원 엔진:

- `bm25`: 키워드 매칭 기반
- `semantic`: 임베딩 코사인 유사도
- `hybrid`: semantic + bm25 점수 결합

현재 운영 기본값:

- 기본 엔진: `bm25`
- 이유: 오프라인/네트워크 제약 환경에서 안정성 우선

안정성 설계:

- semantic/hybrid 실패 시 BM25 fallback
- 설명 실패 시 fallback description으로 인덱스 생성

### 3.5 TTS 및 오디오 합성

설명 텍스트로 클립별 음성을 생성하고, 이를 클립 오디오와 믹싱한다.

- TTS 생성: `audio/*.wav`
- 믹스: 원본 오디오 + TTS 음성 가중 합성
- 최종 병합: TTS 믹싱 클립들로 하이라이트 재생성

결과적으로 TTS가 성공하면 `Full Highlight Video`는 내레이션이 포함된 결과를 우선 사용한다.

### 3.6 웹 구현 (Gradio)

UI는 3단계 흐름으로 구성된다.

- Step 1: 입력 영상 및 옵션 설정
- Step 2: 처리 진행 상태
- Step 3: 결과/클립 리스트/검색

Step 3 구성:

- 상단: Full Highlight Video
- 좌측: Clip List (썸네일 + 설명)
- 우측: Prompt Search

상호작용:

- Clip List 행 선택 시 팝업 플레이어 재생
- 검색 결과는 팝업 리스트로 표시, 선택 재생 가능

렌더링 안정화:

- 썸네일은 data URI로 변환해 브라우저 경로 이슈 없이 표시

---

## 4. 파일 구조

```text
CALF/
├─ inference/
│  ├─ app.py
│  ├─ enhanced_highlight_generator.py
│  ├─ highlight_generator.py
│  ├─ scene_describer.py
│  ├─ scene_search.py
│  ├─ tts_generator.py
│  ├─ result_writer.py
│  ├─ model_registry.py
│  ├─ path_utils.py
│  ├─ vlm_backends/qwen_vl_backend.py
│  └─ config/models.json
├─ src/
├─ models/
└─ README.md
```

---

## 5. 운영 이슈 및 권장 전략

### 5.1 Windows + 대형 VLM

- 16GB VRAM 환경에서 대형 Qwen2-VL은 로딩이 매우 느리거나 실패 가능
- CPU offload로 완화 가능하지만 처리 시간이 크게 증가

권장:

1. 운영 안정성 우선: BM25 + fallback description 유지
2. 품질 우선: 더 작은/양자화된 VLM 또는 Linux CUDA 환경 사용

### 5.2 검색 결과가 비는 경우

주요 원인:

- 설명 실패로 인덱스 미생성
- semantic 모델 다운로드 실패

현재는 fallback description + BM25 fallback으로 대부분 완화됨.

---

## 6. 출력 산출물

기본 경로: `inference/outputs/highlights`

- `highlight.mp4` / `highlight_tts.mp4`
- `clips/*.mp4`
- `clips_tts/*.mp4`
- `audio/*.wav`, `audio/tts_manifest.json`
- `descriptions/scene_descriptions.json`
- `descriptions/search_index_*`
- `inference_result.json`

---

## 7. 사용법 (Quick Start)

### 7.1 설치

```bash
pip install -r inference/requirements_enhanced.txt
pip install ffmpy numpy
```

`ffmpeg`가 시스템 PATH에 있어야 한다.

### 7.2 웹 UI 실행

```bash
python inference/app.py
```

브라우저 접속: `http://localhost:7860`

### 7.3 CLI 실행

```bash
python inference/enhanced_highlight_generator.py \
  --video_path inference/1_720p.mkv \
  --model_name CALF_benchmark \
  --vlm_model_id qwen2vl_soccer_merged_local \
  --search_engine bm25 \
  --hybrid_alpha 0.7 \
  --enable_tts \
  --language auto
```

### 7.4 모델 설정

모델 레지스트리 파일: `inference/config/models.json`

주요 필드:

- `id`, `label`, `hf_path_or_local_path`
- `dtype`, `device_map`, `quantization`
- `allow_cpu_offload`, `gpu_memory_ratio`, `cpu_memory_gb`

---

## 8. 라이선스

Apache v2.0 (`LICENSE` 참조)
