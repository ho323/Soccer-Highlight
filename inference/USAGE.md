# Enhanced Pipeline Usage

## 1. Install dependencies

```bash
pip install -r inference/requirements_enhanced.txt
```

## 2. Run full pipeline (video upload equivalent in CLI)

```bash
python inference/enhanced_highlight_generator.py \
  --video_path inference/1_720p.mkv \
  --model_name CALF_benchmark \
  --vlm_model_id qwen3_vl_7b_default \
  --search_engine hybrid \
  --hybrid_alpha 0.7 \
  --enable_tts \
  --tts_model_id Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --language auto
```

Outputs are saved under `inference/outputs/highlights`:

- `highlight.mp4`
- `descriptions/scene_descriptions.json`
- `audio/tts_manifest.json` and clip-level `audio/clip_XXX.wav`
- `inference_result.json`

## 3. Search modes

```bash
python inference/enhanced_highlight_generator.py \
  --video_path inference/1_720p.mkv \
  --search_query "goal scene" \
  --search_engine semantic
```

```bash
python inference/enhanced_highlight_generator.py \
  --video_path inference/1_720p.mkv \
  --search_query "goal scene" \
  --search_engine bm25
```

```bash
python inference/enhanced_highlight_generator.py \
  --video_path inference/1_720p.mkv \
  --search_query "goal scene" \
  --search_engine hybrid \
  --hybrid_alpha 0.7 \
  --generate_highlight
```

## 4. Gradio UI

```bash
python inference/app.py
```

UI tabs:

- `Pipeline`: upload video, select VLM model, search engine, language, optional TTS
- `Search`: semantic / bm25 / hybrid query with highlight generation
- `Scene Browser`: inspect saved descriptions and clip previews

## 5. Model hot-swap

Edit `inference/config/models.json` to add a fine-tuned Qwen3.5 entry:

```json
{
  "id": "qwen3_5_ft_v1",
  "label": "Qwen3.5 Fine-Tuned v1",
  "model_type": "qwen_vl",
  "hf_path_or_local_path": "your-org/your-qwen3.5-ft-model",
  "dtype": "float16",
  "device_map": "auto",
  "enabled": true
}
```

Restart `inference/app.py`, then select it in the `VLM Model` dropdown.
