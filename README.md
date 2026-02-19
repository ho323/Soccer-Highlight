# CALF Soccer Highlight Pipeline (Qwen VLM + TTS + Search)

축구 영상에서 하이라이트를 자동 생성하는 프로젝트입니다.

- CALF로 이벤트 시점 탐지 (Goal, Foul, Corner 등 17종)
- 이벤트 주변 클립 추출 + 하이라이트 병합
- Qwen VLM으로 장면 설명 생성
- 설명 텍스트 기반 검색 (`semantic / bm25 / hybrid`)
- 설명 결과를 TTS(`Qwen/Qwen3-TTS-12Hz-0.6B-Base`)로 클립별 음성 생성
- 파인튜닝된 Qwen 모델을 레지스트리로 바로 교체 가능

## 1. 프로젝트 한눈에 보기

입력: 축구 영상 1개  
출력:

- `highlight.mp4`
- `descriptions/scene_descriptions.json`
- `descriptions/search_index_*`
- `audio/clip_XXX.wav`, `audio/tts_manifest.json` (옵션)
- `inference_result.json` (런타임 결과 표준 JSON)

핵심 실행 경로:

- Gradio UI: `inference/app.py`
- CLI 파이프라인: `inference/enhanced_highlight_generator.py`

## 2. 폴더 구조

```text
CALF/
├─ inference/
│  ├─ app.py
│  ├─ enhanced_highlight_generator.py
│  ├─ scene_describer.py
│  ├─ scene_search.py
│  ├─ tts_generator.py
│  ├─ result_writer.py
│  ├─ model_registry.py
│  ├─ config/models.json
│  └─ requirements_enhanced.txt
├─ src/                        # 기존 CALF 학습/평가 코드
├─ train_hybrid_ground_truth_final.json
├─ models/                     # 로컬 모델/체크포인트 (git ignore 권장)
└─ README.md
```

## 3. 빠른 시작

### 3.1 환경 준비

```bash
pip install -r inference/requirements_enhanced.txt
pip install ffmpy numpy
```

주의:

- `ffmpeg`가 시스템에 설치되어 있어야 클립 추출/병합이 동작합니다.
- GPU 환경에서 VLM/TTS가 훨씬 빠릅니다.

### 3.2 Gradio UI 실행

```bash
python inference/app.py
```

브라우저에서 `http://localhost:7860` 접속.

### 3.3 CLI 실행

```bash
python inference/enhanced_highlight_generator.py \
  --video_path inference/1_720p.mkv \
  --model_name CALF_benchmark \
  --vlm_model_id qwen2_vl_awq_default \
  --search_engine hybrid \
  --hybrid_alpha 0.7 \
  --enable_tts \
  --tts_model_id Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --language auto
```

## 4. 모델 교체 (파인튜닝 Qwen 즉시 적용)

모델 목록 파일: `inference/config/models.json`

예시:

```json
{
  "models": [
    {
      "id": "qwen2_vl_awq_default",
      "label": "Qwen2-VL-7B-Instruct-AWQ (Default)",
      "model_type": "qwen_vl",
      "hf_path_or_local_path": "Qwen/Qwen2-VL-7B-Instruct-AWQ",
      "dtype": "float16",
      "device_map": "auto",
      "enabled": true
    },
    {
      "id": "qwen3_5_ft_v1",
      "label": "Qwen3.5 Fine-Tuned v1",
      "model_type": "qwen_vl",
      "hf_path_or_local_path": "YOUR_ORG/YOUR_QWEN3_5_FT_MODEL",
      "dtype": "float16",
      "device_map": "auto",
      "enabled": true
    }
  ]
}
```

UI 재시작 후 `VLM Model` 드롭다운에서 선택하면 됩니다.

## 5. 검색 엔진 옵션

- `semantic`: 임베딩 기반 의미 검색
- `bm25`: 키워드 매칭 기반 검색
- `hybrid`: 의미 + 키워드 결합 (`hybrid_alpha`)

CLI 예시:

```bash
python inference/enhanced_highlight_generator.py \
  --video_path inference/1_720p.mkv \
  --search_query "goal scene" \
  --search_engine hybrid \
  --hybrid_alpha 0.7 \
  --generate_highlight
```

## 6. 학습 데이터 JSON과 런타임 JSON

- 학습/파인튜닝 데이터: `train_hybrid_ground_truth_final.json`
- 추론 결과 표준: `inference/outputs/highlights/inference_result.json`

즉, 학습 스키마와 런타임 스키마를 분리해서 운영합니다.

## 7. GitHub 업로드 가이드

이미 `.gitignore`에 아래가 반영되어 있습니다:

- 캐시/가상환경/IDE 파일
- 대용량 산출물 (`inference/outputs`, `outputs`)
- 영상/오디오 (`*.mp4`, `*.mkv`, `*.wav`)
- 모델 체크포인트 (`*.pth`, `*.bin`, `*.safetensors`)

권장 업로드 대상:

- 소스 코드 (`src`, `inference`)
- 설정 파일 (`inference/config/models.json`)
- 문서 (`README.md`, `LICENSE`, `AUTHORS`)
- 샘플/스키마 설명용 JSON (필요 시)

## 8. 라이선스

Apache v2.0  
`LICENSE` 참고.
