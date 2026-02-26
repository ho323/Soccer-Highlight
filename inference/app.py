import os
import base64
from typing import Dict, List, Optional

import cv2

import gradio as gr

from enhanced_highlight_generator import EnhancedHighlightGenerator
from highlight_generator import get_available_events
from model_registry import ModelRegistry
from path_utils import default_highlight_output_dir, default_model_registry_path
from scene_search import SceneSearchEngine

_generator: Optional[EnhancedHighlightGenerator] = None
_search_engine: Optional[SceneSearchEngine] = None
_registry: Optional[ModelRegistry] = None


APP_CSS = """
#clip-modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.72);
  z-index: 1200;
  overflow: auto;
  padding: 28px 16px;
}
#clip-modal-card {
  width: min(960px, 92vw);
  max-height: 90vh;
  overflow: auto;
  background: #111827;
  border: 1px solid #374151;
  border-radius: 12px;
  padding: 14px;
  margin: 0 auto;
}
#search-modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.72);
  z-index: 1190;
  overflow: auto;
  padding: 28px 16px;
}
#search-modal-card {
  width: min(1200px, 95vw);
  max-height: 90vh;
  overflow: auto;
  background: #111827;
  border: 1px solid #374151;
  border-radius: 12px;
  padding: 14px;
  margin: 0 auto;
}
"""


def _get_output_dir() -> str:
    return default_highlight_output_dir()


def _get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(registry_path=default_model_registry_path())
    return _registry


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


def _page_updates(page: int):
    page = 1 if page not in (1, 2, 3) else page
    titles = {
        1: "## Step 1/3: Video Input",
        2: "## Step 2/3: Processing (Clip/TTS/Highlight)",
        3: "## Step 3/3: Results, Thumbnails, and Prompt Search",
    }
    return (
        gr.update(value=titles[page]),
        gr.update(visible=page == 1),
        gr.update(visible=page == 2),
        gr.update(visible=page == 3),
    )


def show_input_page():
    return _page_updates(1)


def show_loading_page():
    return _page_updates(2)


def show_results_page():
    return _page_updates(3)


def toggle_engine_panel(is_open: bool):
    next_open = not bool(is_open)
    return next_open, gr.update(open=next_open)


def open_gallery_clip_popup(gallery_clip_paths_state: List[str], evt: gr.SelectData):
    if not gallery_clip_paths_state:
        return "Clip Preview", None, gr.update(visible=False)

    idx = evt.index
    if isinstance(idx, (tuple, list)):
        idx = idx[0]
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return "Clip Preview", None, gr.update(visible=False)

    if idx < 0 or idx >= len(gallery_clip_paths_state):
        return "Clip Preview", None, gr.update(visible=False)

    clip_path = gallery_clip_paths_state[idx]
    if not clip_path or not os.path.exists(clip_path):
        return "Clip Preview", None, gr.update(visible=False)

    return f"Clip Preview #{idx + 1}", clip_path, gr.update(visible=True)


def close_clip_popup():
    return gr.update(visible=False), None


def reset_clip_popup():
    return "### Clip Preview", None, gr.update(visible=False)


def open_clip_popup_from_table(gallery_clip_paths_state: List[str], evt: gr.SelectData):
    idx = evt.index
    if isinstance(idx, (tuple, list)):
        idx = idx[0]
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return "Clip Preview", None, gr.update(visible=False)

    if idx < 0 or idx >= len(gallery_clip_paths_state):
        return "Clip Preview", None, gr.update(visible=False)
    clip_path = gallery_clip_paths_state[idx]
    if not clip_path or not os.path.exists(clip_path):
        return "Clip Preview", None, gr.update(visible=False)
    return f"Clip Preview #{idx + 1}", clip_path, gr.update(visible=True)


def open_search_modal():
    return gr.update(visible=True)


def close_search_modal():
    return gr.update(visible=False)


def _format_mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    mm = s // 60
    ss = s % 60
    return f"{mm:02d}:{ss:02d}"


def _write_thumbnail(clip_path: str, output_path: str) -> bool:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return False
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    return bool(cv2.imwrite(output_path, frame))


def _thumbnail_markdown_data_uri(image_path: str) -> str:
    if not os.path.exists(image_path):
        return ""
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"![thumb](data:image/jpeg;base64,{b64})"
    except OSError:
        return ""


def _thumbnail_needs_refresh(clip_path: str, thumb_path: str) -> bool:
    if not os.path.exists(thumb_path):
        return True
    return os.path.getmtime(thumb_path) < os.path.getmtime(clip_path)


def _build_result_assets(result: Dict):
    events = result.get("events", []) or []
    descriptions = result.get("descriptions", []) or []
    clip_paths = result.get("clip_paths", []) or []
    tts_rows = result.get("tts", []) or []

    thumbs_dir = os.path.join(_get_output_dir(), "thumbnails")
    os.makedirs(thumbs_dir, exist_ok=True)

    clip_rows = []
    gallery_clip_paths = []
    scene_lines = []
    count = len(clip_paths)
    for i in range(count):
        clip_path = clip_paths[i]
        desc = descriptions[i] if i < len(descriptions) else {}
        event = events[i] if i < len(events) else {}
        event_type = desc.get("event_type") or event.get("event_type", "Unknown")
        timestamp = float(desc.get("timestamp", event.get("timestamp", 0.0)))
        confidence = float(desc.get("confidence", event.get("confidence", 0.0)))
        description_text = desc.get("description", "")
        time_mmss = _format_mmss(timestamp)

        thumb_path = os.path.join(thumbs_dir, f"clip_{i:04d}.jpg")
        thumb_ok = False
        if os.path.exists(clip_path):
            if _thumbnail_needs_refresh(clip_path, thumb_path):
                thumb_ok = _write_thumbnail(clip_path, thumb_path)
            else:
                thumb_ok = True
        if thumb_ok:
            snippet = description_text if description_text else "No description"
            row_text = f"[{i + 1}] {time_mmss} | {event_type}\n{snippet}"
            thumb_md = _thumbnail_markdown_data_uri(thumb_path)
            clip_rows.append([thumb_md, row_text])
            gallery_clip_paths.append(clip_path)

        scene_lines.append(
            f"[{i + 1}] {time_mmss} | {event_type} (conf: {confidence:.3f})\n  {description_text}"
        )

    tts_lines = []
    for i, row in enumerate(tts_rows):
        lang = row.get("language", "")
        audio_path = row.get("audio_path", "")
        text = row.get("text", "")
        tts_lines.append(f"[{i + 1}] {lang} | {audio_path}\n  {text[:180]}")

    scene_text = "\n\n".join(scene_lines) if scene_lines else "No extracted clips/descriptions."
    tts_text = "\n\n".join(tts_lines) if tts_lines else "No TTS results."
    return clip_rows, scene_text, tts_text, gallery_clip_paths


def search_scenes_for_popup(query: str, top_k: int, min_score: float, search_engine: str, hybrid_alpha: float):
    if not query.strip():
        return "Please enter a search query.", [], [], gr.update(visible=False)

    engine = _ensure_search_engine(search_engine=search_engine, hybrid_alpha=hybrid_alpha)
    results = engine.search(query, top_k=top_k, min_score=min_score)
    if not results:
        return "No similar clips found.", [], [], gr.update(visible=False)

    thumbs_dir = os.path.join(_get_output_dir(), "thumbnails", "search")
    os.makedirs(thumbs_dir, exist_ok=True)

    gallery_items = []
    clip_paths = []
    for i, row in enumerate(results):
        clip_path = row.get("clip_path", "")
        if not clip_path or not os.path.exists(clip_path):
            continue
        thumb_path = os.path.join(thumbs_dir, f"search_{i:04d}.jpg")
        if _thumbnail_needs_refresh(clip_path, thumb_path):
            _write_thumbnail(clip_path, thumb_path)
        if not os.path.exists(thumb_path):
            continue
        time_mmss = _format_mmss(float(row.get("timestamp", 0.0)))
        event_type = row.get("event_type", "Unknown")
        score = float(row.get("search_score", 0.0))
        caption = f"{time_mmss} | {event_type} | score {score:.3f}"
        gallery_items.append((thumb_path, caption))
        clip_paths.append(clip_path)

    if not gallery_items:
        return "No playable clips in search results.", [], [], gr.update(visible=False)
    return f"Found {len(gallery_items)} similar clips.", clip_paths, gallery_items, gr.update(visible=True)


def run_pipeline_for_wizard(
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
    global _generator, _search_engine

    if video_file is None:
        return (
            "Please upload a video.",
            None,
            gr.update(value=[]),
            [],
        )

    vlm_model_id = _model_label_to_id(model_label)
    if not vlm_model_id:
        return (
            f"Unknown VLM model label: {model_label}",
            None,
            gr.update(value=[]),
            [],
        )

    event_types = event_types_selected if event_types_selected else None
    max_clips_val = max_clips if max_clips > 0 else None
    progress(0.1, desc="Initializing...")

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
    progress(0.3, desc="Extracting clips and generating highlight...")

    try:
        result = _generator.run_full_pipeline(
            confidence_threshold=confidence_threshold,
            event_types=event_types,
            before_event=before_event,
            after_event=after_event,
            max_clips=max_clips_val,
            skip_description=skip_description,
        )
    except Exception as exc:
        progress(1.0, desc="Failed")
        return (
            "Pipeline failed",
            None,
            gr.update(value=[]),
            [],
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
    for warning in result.get("warnings", []):
        summary_lines.append(f"Warning: {warning}")

    _search_engine = None

    clip_rows, _scene_text, _tts_text, gallery_clip_paths = _build_result_assets(result)
    highlight_path = result.get("highlight_path", "")
    highlight_video = highlight_path if highlight_path and os.path.exists(highlight_path) else None
    status = "Pipeline completed" if highlight_video else "Pipeline failed"
    return (
        status,
        highlight_video,
        gr.update(value=clip_rows),
        gallery_clip_paths,
    )


def create_ui() -> gr.Blocks:
    all_events = get_available_events()
    registry = _get_registry()
    model_choices = registry.get_model_choices()
    default_model = model_choices[0] if model_choices else ""

    with gr.Blocks(
        title="Soccer Highlight Generator + Qwen/TTS Search",
        theme=gr.themes.Soft(),
        css=APP_CSS,
    ) as demo:
        gr.Markdown("# Soccer Highlight Generator with Pluggable Qwen VLM + TTS")
        page_title = gr.Markdown("## Step 1/3: Video Input")
        search_clip_paths_state = gr.State([])
        search_result_paths_state = gr.State([])
        engine_panel_open_state = gr.State(False)
        gallery_clip_paths_state = gr.State([])
        pipeline_status_state = gr.State("")

        with gr.Column(visible=True) as page1:
            gr.Markdown("Upload a video, set options, and press Start Processing.")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Input Video", sources=["upload"])
                with gr.Column(scale=1):
                    confidence_slider = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence Threshold")
                    event_checkboxes = gr.CheckboxGroup(choices=all_events, label="Event Type Filter (empty = all)")
                    with gr.Row():
                        before_slider = gr.Slider(1, 15, value=5, step=1, label="Before Event (sec)")
                        after_slider = gr.Slider(1, 15, value=5, step=1, label="After Event (sec)")
                    max_clips_input = gr.Number(value=0, label="Max Clips (0 = no limit)", precision=0)
                    skip_desc_checkbox = gr.Checkbox(label="Skip Description", value=False)
                    model_dropdown = gr.Dropdown(choices=model_choices, value=default_model, label="VLM Model")
                    enable_tts_checkbox = gr.Checkbox(label="Enable TTS", value=True)
                    language_dropdown = gr.Dropdown(
                        choices=["auto", "ko", "en"],
                        value="auto",
                        label="Language",
                    )
                    engine_dropdown = gr.Dropdown(
                        choices=["semantic", "bm25", "hybrid"],
                        value="bm25",
                        label="Search Engine",
                    )
                    alpha_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Hybrid Alpha")
                    start_btn = gr.Button("Start Processing", variant="primary")

        with gr.Column(visible=False) as page2:
            gr.Markdown("### Processing...")
            gr.Markdown("- Clip extraction\n- TTS generation (if enabled)\n- Highlight generation\n- Search index build")
            loading_status = gr.Textbox(label="Processing Status", interactive=False, value="Waiting...")
            go_back_input_btn = gr.Button("Back To Input")

        with gr.Column(visible=False) as page3:
            pipeline_video = gr.Video(label="Full Highlight Video", height=500)
            with gr.Row():
                with gr.Column(scale=3):
                    clip_table = gr.Dataframe(
                        headers=["", "Clip List"],
                        datatype=["markdown", "str"],
                        value=[],
                        interactive=False,
                        wrap=True,
                        row_count=(0, "dynamic"),
                        col_count=(2, "fixed"),
                        label="Clip List",
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Search Engine")
                    search_query = gr.Textbox(label="Prompt", placeholder="e.g. goal scene, penalty kick")
                    with gr.Row():
                        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top K")
                        min_score_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Min Score")
                    toggle_engine_btn = gr.Button("Search Engine Settings")
                    with gr.Accordion("Engine Options", open=False) as engine_accordion:
                        search_engine_dropdown = gr.Dropdown(
                            choices=["semantic", "bm25", "hybrid"],
                            value="bm25",
                            label="Search Engine",
                        )
                        search_alpha_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Hybrid Alpha")
                    search_btn = gr.Button("Find Similar Clips", variant="primary")
                    search_status = gr.Textbox(label="Search Status", interactive=False)
                    highlight_btn = gr.Button("Generate Search Highlight")
                    search_highlight_status = gr.Textbox(label="Search Highlight Status", interactive=False)
                    search_highlight_video = gr.Video(label="Search Highlight Video")
                    restart_btn = gr.Button("Process Another Video")

            with gr.Column(visible=False, elem_id="clip-modal") as clip_modal:
                with gr.Column(elem_id="clip-modal-card"):
                    clip_modal_title = gr.Markdown("### Clip Preview")
                    clip_modal_video = gr.Video(label="Popup Clip Player")
                    close_modal_btn = gr.Button("Close")

            with gr.Column(visible=False, elem_id="search-modal") as search_modal:
                with gr.Column(elem_id="search-modal-card"):
                    gr.Markdown("### Similar Clips")
                    search_result_gallery = gr.Gallery(label="Similar Clip List", columns=4, rows=2, height=520)
                    close_search_modal_btn = gr.Button("Close Search List")

        start_btn.click(
            fn=show_loading_page,
            outputs=[page_title, page1, page2, page3],
        ).then(
            fn=reset_clip_popup,
            outputs=[clip_modal_title, clip_modal_video, clip_modal],
        ).then(
            fn=run_pipeline_for_wizard,
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
            outputs=[
                pipeline_status_state,
                pipeline_video,
                clip_table,
                gallery_clip_paths_state,
            ],
        ).then(
            fn=lambda status: status,
            inputs=[pipeline_status_state],
            outputs=[loading_status],
        ).then(
            fn=show_results_page,
            outputs=[page_title, page1, page2, page3],
        ).then(
            fn=close_search_modal,
            outputs=[search_modal],
        )

        clip_table.select(
            fn=open_clip_popup_from_table,
            inputs=[gallery_clip_paths_state],
            outputs=[clip_modal_title, clip_modal_video, clip_modal],
        )
        close_modal_btn.click(
            fn=close_clip_popup,
            outputs=[clip_modal, clip_modal_video],
        )

        go_back_input_btn.click(
            fn=show_input_page,
            outputs=[page_title, page1, page2, page3],
        ).then(
            fn=reset_clip_popup,
            outputs=[clip_modal_title, clip_modal_video, clip_modal],
        )

        restart_btn.click(
            fn=show_input_page,
            outputs=[page_title, page1, page2, page3],
        ).then(
            fn=reset_clip_popup,
            outputs=[clip_modal_title, clip_modal_video, clip_modal],
        ).then(
            fn=close_search_modal,
            outputs=[search_modal],
        )

        toggle_engine_btn.click(
            fn=toggle_engine_panel,
            inputs=[engine_panel_open_state],
            outputs=[engine_panel_open_state, engine_accordion],
        )

        search_btn.click(
            fn=search_scenes_for_popup,
            inputs=[search_query, top_k_slider, min_score_slider, search_engine_dropdown, search_alpha_slider],
            outputs=[search_status, search_result_paths_state, search_result_gallery, search_modal],
        )
        highlight_btn.click(
            fn=generate_search_highlight,
            inputs=[search_query, top_k_slider, min_score_slider, search_engine_dropdown, search_alpha_slider],
            outputs=[search_highlight_status, search_highlight_video],
        )

        search_result_gallery.select(
            fn=open_gallery_clip_popup,
            inputs=[search_result_paths_state],
            outputs=[clip_modal_title, clip_modal_video, clip_modal],
        )
        close_search_modal_btn.click(
            fn=close_search_modal,
            outputs=[search_modal],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
