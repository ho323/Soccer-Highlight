import os
from typing import Dict, List, Optional

import gradio as gr

from enhanced_highlight_generator import EnhancedHighlightGenerator
from highlight_generator import get_available_events
from model_registry import ModelRegistry
from scene_describer import SceneDescriber
from scene_search import SceneSearchEngine

_generator: Optional[EnhancedHighlightGenerator] = None
_search_engine: Optional[SceneSearchEngine] = None
_descriptions: Optional[List[Dict]] = None


def _get_output_dir() -> str:
    return "inference/outputs/highlights"


def _get_registry() -> ModelRegistry:
    return ModelRegistry(registry_path="inference/config/models.json")


def _model_label_to_id(model_label: str) -> Optional[str]:
    return _get_registry().resolve_id_from_label(model_label)


def _ensure_search_engine(search_engine: str, hybrid_alpha: float) -> SceneSearchEngine:
    global _search_engine
    if _search_engine is None:
        _search_engine = SceneSearchEngine(
            output_dir=_get_output_dir(),
            engine_type=search_engine,
            hybrid_alpha=hybrid_alpha,
        )
    else:
        _search_engine.set_engine(search_engine, hybrid_alpha=hybrid_alpha)
    _search_engine.load_index()
    return _search_engine


def _load_descriptions(vlm_model_id: Optional[str] = None, language: str = "auto") -> List[Dict]:
    global _descriptions
    if _descriptions is None:
        describer = SceneDescriber(
            output_dir=_get_output_dir(),
            model_id=vlm_model_id,
            language=language,
        )
        _descriptions = describer.load_descriptions()
    return _descriptions or []


def run_pipeline(
    video_file,
    confidence_threshold: float,
    event_types_selected: List[str],
    before_event: float,
    after_event: float,
    max_clips: int,
    skip_description: bool,
    model_label: str,
    enable_tts: bool,
    language: str,
    search_engine: str,
    hybrid_alpha: float,
    progress=gr.Progress(),
):
    global _generator, _search_engine, _descriptions

    if video_file is None:
        return "Please upload a video.", None, ""

    vlm_model_id = _model_label_to_id(model_label)
    if not vlm_model_id:
        return f"Unknown VLM model label: {model_label}", None, "Failed"

    event_types = event_types_selected if event_types_selected else None
    max_clips_val = max_clips if max_clips > 0 else None
    progress(0.1, desc="Initializing pipeline...")

    _generator = EnhancedHighlightGenerator(
        video_path=video_file,
        output_dir=_get_output_dir(),
        model_name="CALF_benchmark",
        vlm_model_id=vlm_model_id,
        enable_tts=enable_tts,
        language=language,
        search_engine=search_engine,
        hybrid_alpha=hybrid_alpha,
    )
    progress(0.3, desc="Running extraction and description...")

    result = _generator.run_full_pipeline(
        confidence_threshold=confidence_threshold,
        event_types=event_types,
        before_event=before_event,
        after_event=after_event,
        max_clips=max_clips_val,
        skip_description=skip_description,
    )
    progress(1.0, desc="Done")

    summary_lines = [
        f"Detected events: {len(result.get('events', []))}",
        f"Extracted clips: {len(result.get('clip_paths', []))}",
        f"Descriptions: {len(result.get('descriptions', []))}",
        f"TTS entries: {len(result.get('tts', []))}",
        f"Search index ready: {result.get('index_ready', False)}",
        f"Result JSON: {result.get('inference_result_path', '')}",
    ]

    _search_engine = None
    _descriptions = None

    highlight_path = result.get("highlight_path", "")
    if highlight_path and os.path.exists(highlight_path):
        return "\n".join(summary_lines), highlight_path, "Pipeline completed"
    return "\n".join(summary_lines), None, "Pipeline failed"


def search_scenes(query: str, top_k: int, min_score: float, search_engine: str, hybrid_alpha: float):
    if not query.strip():
        return "Please enter a search query.", [], None

    engine = _ensure_search_engine(search_engine=search_engine, hybrid_alpha=hybrid_alpha)
    results = engine.search(query, top_k=top_k, min_score=min_score)
    if not results:
        return "No search results.", [], None

    lines = []
    clip_paths = []
    for row in results:
        lines.append(
            f"#{row['search_rank']} [{row['search_engine']}:{row['search_score']:.3f}] "
            f"{row['event_type']} @ {row['timestamp']:.1f}s\n  {row['description']}"
        )
        clip_paths.append(row.get("clip_path", ""))

    preview = clip_paths[0] if clip_paths and os.path.exists(clip_paths[0]) else None
    return "\n\n".join(lines), clip_paths, preview


def generate_search_highlight(query: str, top_k: int, min_score: float, search_engine: str, hybrid_alpha: float):
    if _generator is None:
        return "Run pipeline first so generator state exists.", None
    if not query.strip():
        return "Please enter a search query.", None
    path = _generator.generate_search_highlight(
        query=query,
        top_k=top_k,
        min_score=min_score,
        search_engine=search_engine,
        hybrid_alpha=hybrid_alpha,
    )
    if path and os.path.exists(path):
        return "Search highlight generated.", path
    return "Failed to generate search highlight.", None


def preview_clip(clip_paths_state, index: int):
    if not clip_paths_state or index < 0 or index >= len(clip_paths_state):
        return None
    path = clip_paths_state[index]
    return path if os.path.exists(path) else None


def browse_scenes(event_type_filter: str, model_label: str, language: str):
    model_id = _model_label_to_id(model_label)
    descriptions = _load_descriptions(vlm_model_id=model_id, language=language)
    if not descriptions:
        return "No saved scene descriptions. Run pipeline first.", None

    if event_type_filter and event_type_filter != "All":
        descriptions = [d for d in descriptions if d.get("event_type") == event_type_filter]
    if not descriptions:
        return f"No scenes found for event type '{event_type_filter}'.", None

    lines = []
    for i, desc in enumerate(descriptions):
        lines.append(
            f"[{i + 1}] {desc['event_type']} @ {desc['timestamp']:.1f}s "
            f"(confidence: {desc.get('confidence', 0.0):.3f})\n    {desc.get('description', 'N/A')}"
        )

    preview = descriptions[0].get("clip_path")
    preview = preview if preview and os.path.exists(preview) else None
    return "\n\n".join(lines), preview


def browse_clip_by_index(event_type_filter: str, clip_index: int, model_label: str, language: str):
    model_id = _model_label_to_id(model_label)
    descriptions = _load_descriptions(vlm_model_id=model_id, language=language)
    if event_type_filter and event_type_filter != "All":
        descriptions = [d for d in descriptions if d.get("event_type") == event_type_filter]

    idx = int(clip_index) - 1
    if 0 <= idx < len(descriptions):
        path = descriptions[idx].get("clip_path", "")
        if os.path.exists(path):
            return path
    return None


def create_ui() -> gr.Blocks:
    all_events = get_available_events()
    registry = _get_registry()
    model_choices = registry.get_model_choices()
    default_model = model_choices[0] if model_choices else ""

    with gr.Blocks(title="Soccer Highlight Generator + Qwen/TTS Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Soccer Highlight Generator with Pluggable Qwen VLM + TTS")
        gr.Markdown("Upload video -> detect events -> describe scenes -> build search index -> optional clip-level TTS.")

        with gr.Tab("Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Input Video", sources=["upload"])
                    confidence_slider = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence Threshold")
                    event_checkboxes = gr.CheckboxGroup(choices=all_events, label="Event Type Filter (empty = all)")
                    with gr.Row():
                        before_slider = gr.Slider(1, 15, value=5, step=1, label="Before Event (sec)")
                        after_slider = gr.Slider(1, 15, value=5, step=1, label="After Event (sec)")
                    max_clips_input = gr.Number(value=0, label="Max Clips (0 = no limit)", precision=0)
                    skip_desc_checkbox = gr.Checkbox(label="Skip Description", value=False)
                    model_dropdown = gr.Dropdown(choices=model_choices, value=default_model, label="VLM Model")
                    enable_tts_checkbox = gr.Checkbox(label="Enable TTS", value=False)
                    language_dropdown = gr.Dropdown(
                        choices=["auto", "ko", "en"],
                        value="auto",
                        label="Language",
                    )
                    engine_dropdown = gr.Dropdown(
                        choices=["semantic", "bm25", "hybrid"],
                        value="semantic",
                        label="Search Engine",
                    )
                    alpha_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Hybrid Alpha")
                    run_btn = gr.Button("Run Pipeline", variant="primary")

                with gr.Column(scale=1):
                    pipeline_status = gr.Textbox(label="Status", interactive=False)
                    pipeline_summary = gr.Textbox(label="Summary", lines=15, interactive=False)
                    pipeline_video = gr.Video(label="Highlight Video")

            run_btn.click(
                fn=run_pipeline,
                inputs=[
                    video_input,
                    confidence_slider,
                    event_checkboxes,
                    before_slider,
                    after_slider,
                    max_clips_input,
                    skip_desc_checkbox,
                    model_dropdown,
                    enable_tts_checkbox,
                    language_dropdown,
                    engine_dropdown,
                    alpha_slider,
                ],
                outputs=[pipeline_summary, pipeline_video, pipeline_status],
            )

        with gr.Tab("Search"):
            search_clip_paths_state = gr.State([])

            with gr.Row():
                with gr.Column(scale=1):
                    search_query = gr.Textbox(label="Search Query", placeholder="e.g. goal scene, penalty kick")
                    with gr.Row():
                        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top K")
                        min_score_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Min Score")
                    search_engine_dropdown = gr.Dropdown(
                        choices=["semantic", "bm25", "hybrid"],
                        value="semantic",
                        label="Search Engine",
                    )
                    search_alpha_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Hybrid Alpha")
                    search_btn = gr.Button("Search", variant="primary")
                    highlight_btn = gr.Button("Generate Search Highlight")

                with gr.Column(scale=1):
                    search_results_text = gr.Textbox(label="Search Results", lines=12, interactive=False)
                    search_preview = gr.Video(label="Clip Preview")
                    search_highlight_status = gr.Textbox(label="Status", interactive=False)
                    search_highlight_video = gr.Video(label="Search Highlight")

            with gr.Row():
                clip_index_slider = gr.Slider(0, 19, value=0, step=1, label="Clip Index (0-based)")
                preview_btn = gr.Button("Preview Clip")

            search_btn.click(
                fn=search_scenes,
                inputs=[search_query, top_k_slider, min_score_slider, search_engine_dropdown, search_alpha_slider],
                outputs=[search_results_text, search_clip_paths_state, search_preview],
            )
            highlight_btn.click(
                fn=generate_search_highlight,
                inputs=[search_query, top_k_slider, min_score_slider, search_engine_dropdown, search_alpha_slider],
                outputs=[search_highlight_status, search_highlight_video],
            )
            preview_btn.click(
                fn=preview_clip,
                inputs=[search_clip_paths_state, clip_index_slider],
                outputs=[search_preview],
            )

        with gr.Tab("Scene Browser"):
            with gr.Row():
                with gr.Column(scale=1):
                    browser_filter = gr.Dropdown(choices=["All"] + all_events, value="All", label="Event Type Filter")
                    browser_model_dropdown = gr.Dropdown(choices=model_choices, value=default_model, label="VLM Model")
                    browser_lang_dropdown = gr.Dropdown(choices=["auto", "ko", "en"], value="auto", label="Language")
                    browse_btn = gr.Button("Show Scene List", variant="primary")
                    browser_clip_index = gr.Number(value=1, label="Clip Number (1-based)", precision=0)
                    browse_preview_btn = gr.Button("Open Clip")

                with gr.Column(scale=1):
                    browser_text = gr.Textbox(label="Scene List", lines=15, interactive=False)
                    browser_video = gr.Video(label="Clip Preview")

            browse_btn.click(
                fn=browse_scenes,
                inputs=[browser_filter, browser_model_dropdown, browser_lang_dropdown],
                outputs=[browser_text, browser_video],
            )
            browse_preview_btn.click(
                fn=browse_clip_by_index,
                inputs=[browser_filter, browser_clip_index, browser_model_dropdown, browser_lang_dropdown],
                outputs=[browser_video],
            )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
