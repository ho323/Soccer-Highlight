# CALF Soccer Highlight Pipeline (Qwen VLM + TTS + Search)

축구 영상에서 하이라이트를 자동 생성하는 파이프라인입니다.

- CALF 이벤트 탐지 (17개 이벤트)
- 이벤트 기반 클립 추출 + 하이라이트 병합
- Qwen VLM 장면 설명 (가능 시)
- 검색 엔진 (`bm25`, `semantic`, `hybrid`)
- TTS 생성 + 클립 오디오 믹싱 + TTS 하이라이트 생성
- Gradio 단계형 UI (Step1 입력 → Step2 처리 → Step3 결과)

## 1. Project Structure

```text
CALF/
├─ inference/
│  ├─ app.py
│  ├─ enhanced_highlight_generator.py
│  ├─ highlight_generator.py
│  ├─ scene_describer.py
│  ├─ scene_search.py
│  ├─ tts_generator.py
│  ├─ vlm_backends/qwen_vl_backend.py
│  ├─ config/models.json
│  └─ path_utils.py
├─ src/
├─ models/
└─ README.md
```

## 2. Run

### 2.1 Install

```bash
pip install -r inference/requirements_enhanced.txt
pip install ffmpy numpy
```

`ffmpeg`가 PATH에 있어야 합니다.

### 2.2 UI

```bash
python inference/app.py
```

브라우저: `http://localhost:7860`

## 3. Current UI Flow

- **Step 1**: 비디오 업로드 + 파라미터 설정
- **Step 2**: 처리 대기
- **Step 3**:
  - 상단: Full Highlight Video
  - 좌측: Clip List (썸네일 + 설명 행 리스트, 클릭 시 팝업 재생)
  - 우측: 검색 엔진 (프롬프트 검색, 유사 클립 팝업)

## 4. Search / TTS Behavior

- 기본 검색 엔진은 `bm25`로 설정되어 있습니다.
- VLM 설명이 실패해도 이벤트 기반 fallback description으로 검색 인덱스를 생성합니다.
- TTS가 생성되면 클립에 믹싱 후 `highlight_tts.mp4`를 우선 결과로 사용합니다.

## 5. VLM Model Config (`inference/config/models.json`)

예시 필드:

- `hf_path_or_local_path`: 로컬 경로 또는 HF 모델 ID
- `dtype`: `float16` 등
- `device_map`: `auto`, `cuda:0`, `cpu`
- `quantization`: `""`, `"8bit"`, `"4bit"`
- `allow_cpu_offload`: `true/false`
- `gpu_memory_ratio`: GPU 할당 비율
- `cpu_memory_gb`: CPU 오프로딩 메모리 예산

## 6. Important Notes (Windows)

- 대형 로컬 Qwen2-VL 모델은 16GB VRAM 환경에서 매우 느리거나 실패할 수 있습니다.
- 8bit는 환경/런타임 조합에 따라 실패할 수 있습니다.
- CPU offload 모드에서는 로딩 시간이 매우 길 수 있습니다.

## 7. Output Files

기본 출력 디렉터리: `inference/outputs/highlights`

- `highlight.mp4` 또는 `highlight_tts.mp4`
- `clips/*.mp4`
- `clips_tts/*.mp4` (TTS 믹싱 클립)
- `audio/*.wav`, `audio/tts_manifest.json`
- `descriptions/scene_descriptions.json`
- `descriptions/search_index_*`
- `inference_result.json`

## 8. Git

대용량 산출물/모델 파일은 `.gitignore`로 제외되어 있습니다.

## 9. License

Apache v2.0 (`LICENSE` 참고)
