from __future__ import annotations

import hmac
import io
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import streamlit as st
import torch
from bert_score import BERTScorer
from PIL import Image, UnidentifiedImageError
from sacrebleu.metrics import BLEU
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import functional as TF

from storage import (
    HISTORY_LIMIT,
    MAX_IMAGE_SIZE_BYTES,
    StorageStatus,
    get_storage_status,
    list_recent_image_evaluations,
    list_recent_text_evaluations,
    resolve_write_password,
    save_image_evaluation,
    save_text_evaluation,
)


APP_TITLE: Final[str] = "Evaluador multimodal de texto e imágenes"
BERTSCORE_MODEL_NAME: Final[str] = "bert-base-multilingual-cased"
CLIP_MODEL_NAME: Final[str] = "openai/clip-vit-base-patch32"
FID_IMAGE_SIZE: Final[tuple[int, int]] = (299, 299)
DEVICE: Final[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class WriteAccessStatus:
    configured: bool
    unlocked: bool
    can_write: bool
    message: str


def parse_references(text: str) -> list[str]:
    """Split references by line and drop empty entries."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def compute_bleu(candidate: str, references: list[str]) -> float:
    """Compute corpus BLEU for one hypothesis against multiple references."""
    if not references:
        raise ValueError("No hay referencias válidas para calcular BLEU.")

    bleu_metric = BLEU()
    references_by_corpus = [[reference] for reference in references]
    return float(bleu_metric.corpus_score([candidate], references_by_corpus).score)


@st.cache_resource(show_spinner=False)
def get_bert_scorer(device_name: str) -> BERTScorer:
    return BERTScorer(
        model_type=BERTSCORE_MODEL_NAME,
        device=device_name,
        rescale_with_baseline=False,
    )


def compute_bertscore(candidate: str, references: list[str]) -> dict[str, float]:
    """Average precision, recall and F1 across all references."""
    if not references:
        raise ValueError("No hay referencias válidas para calcular BERTScore.")

    scorer = get_bert_scorer(str(DEVICE))
    candidates = [candidate] * len(references)

    with torch.inference_mode():
        precision, recall, f1 = scorer.score(candidates, references)

    return {
        "precision": float(precision.mean().detach().cpu().item()),
        "recall": float(recall.mean().detach().cpu().item()),
        "f1": float(f1.mean().detach().cpu().item()),
    }


def load_image(uploaded_file) -> Image.Image:
    """Load an uploaded image and normalize it to RGB."""
    if uploaded_file is None:
        raise ValueError("Debes cargar una imagen válida.")

    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"No se pudo leer el archivo '{uploaded_file.name}' como imagen.") from exc
    except OSError as exc:
        raise ValueError(f"El archivo '{uploaded_file.name}' parece estar dañado o incompleto.") from exc


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a CHW tensor with uint8 values."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return TF.pil_to_tensor(image)


def resize_for_fid(image: Image.Image) -> Image.Image:
    return image.resize(FID_IMAGE_SIZE, Image.Resampling.BICUBIC)


def prepare_fid_batch(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    tensors = [pil_to_tensor(resize_for_fid(image)) for image in images]
    return torch.stack(tensors, dim=0).to(device)


def compute_fid_single_vs_group(target_image: Image.Image, reference_images: list[Image.Image]) -> float:
    """Compute FID for one image against a reference group."""
    if target_image is None:
        raise ValueError("La imagen objetivo no es válida.")
    if len(reference_images) != 2:
        raise ValueError("FID requiere exactamente dos imágenes de referencia para este módulo.")

    try:
        metric = FrechetInceptionDistance(feature=64, normalize=False).to(DEVICE)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FID requiere dependencias adicionales. Verifica que 'torch-fidelity' esté instalado."
        ) from exc

    metric = metric.set_dtype(torch.float64)

    target_batch = prepare_fid_batch([target_image], DEVICE)
    reference_batch = prepare_fid_batch(reference_images, DEVICE)

    with torch.inference_mode():
        metric.update(reference_batch, real=True)
        metric.update(target_batch, real=False)
        fid_value = float(metric.compute().detach().cpu().item())

    if not math.isfinite(fid_value):
        raise ValueError("No fue posible obtener un valor FID finito con las imágenes proporcionadas.")

    return fid_value


@st.cache_resource(show_spinner=False)
def get_clip_metric(device_name: str) -> CLIPScore:
    metric = CLIPScore(model_name_or_path=CLIP_MODEL_NAME)
    return metric.to(device_name)


def compute_clipscore(image: Image.Image, text: str) -> float:
    """Compute CLIPScore between one image and one text."""
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Debes ingresar un texto para calcular CLIPScore.")

    metric = get_clip_metric(str(DEVICE))
    image_tensor = pil_to_tensor(image).to(DEVICE)

    with torch.inference_mode():
        metric.reset()
        clip_value = float(metric(image_tensor, cleaned_text).detach().cpu().item())
        metric.reset()

    return clip_value


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "Sin fecha"
    return value.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def validate_text_inputs(source_text: str, candidate_texts: list[str]) -> tuple[str, list[tuple[str, str]]]:
    cleaned_source = source_text.strip()
    if not cleaned_source:
        raise ValueError("Debes ingresar el texto fuente.")

    cleaned_candidates: list[tuple[str, str]] = []
    for index, text in enumerate(candidate_texts, start=1):
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError(f"Debes ingresar el Texto {index}.")
        cleaned_candidates.append((f"Texto {index}", cleaned_text))

    return cleaned_source, cleaned_candidates


def validate_image_inputs(text: str, uploaded_files: list) -> str:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Debes ingresar el texto para comparar con las imágenes.")
    if any(file is None for file in uploaded_files):
        raise ValueError("Debes cargar las tres imágenes antes de evaluar.")

    size_limit_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            raise ValueError(f"La Imagen {index} no contiene datos.")
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"La Imagen {index} supera el límite de {size_limit_mb} MB.")
    return cleaned_text


def render_text_results(results: list[dict[str, object]]) -> None:
    st.success("Evaluación de texto completada.")

    for result in results:
        bert_results = result["bert_results"]
        with st.container(border=True):
            st.markdown(f"**{result['label']}**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("BLEU", format_metric(float(result["bleu_score"])))
            col2.metric("BERTScore Precision", format_metric(float(bert_results["precision"])))
            col3.metric("BERTScore Recall", format_metric(float(bert_results["recall"])))
            col4.metric("BERTScore F1", format_metric(float(bert_results["f1"])))

    st.info(
        "BLEU alto = mayor similitud textual superficial.\n\n"
        "BERTScore alto = mayor cercanía semántica."
    )


def render_image_results(fid_results: dict[str, float], clip_results: dict[str, float]) -> None:
    st.success("Evaluación de imágenes completada.")

    st.subheader("Resultados FID")
    fid_col1, fid_col2, fid_col3 = st.columns(3)
    fid_col1.metric("FID Imagen 1 vs Imágenes 2 y 3", format_metric(fid_results["fid_1_vs_23"]))
    fid_col2.metric("FID Imagen 2 vs Imágenes 1 y 3", format_metric(fid_results["fid_2_vs_13"]))
    fid_col3.metric("FID Imagen 3 vs Imágenes 1 y 2", format_metric(fid_results["fid_3_vs_12"]))

    st.subheader("Resultados CLIPScore")
    clip_col1, clip_col2, clip_col3 = st.columns(3)
    clip_col1.metric("CLIPScore Imagen 1 vs Texto", format_metric(clip_results["clip_1"]))
    clip_col2.metric("CLIPScore Imagen 2 vs Texto", format_metric(clip_results["clip_2"]))
    clip_col3.metric("CLIPScore Imagen 3 vs Texto", format_metric(clip_results["clip_3"]))

    st.info(
        "Menor FID suele indicar mayor similitud visual respecto al conjunto comparado.\n\n"
        "Mayor CLIPScore suele indicar mejor correspondencia entre imagen y texto."
    )


def render_storage_banner(storage_status: StorageStatus) -> None:
    if storage_status.available:
        st.caption(storage_status.message)
    else:
        st.warning(storage_status.message)


def get_write_access_status(write_password: str | None) -> WriteAccessStatus:
    if not write_password:
        return WriteAccessStatus(
            configured=False,
            unlocked=False,
            can_write=False,
            message="Guardado protegido: configura STORAGE_WRITE_PASSWORD para habilitar escritura en la base.",
        )

    unlocked = bool(st.session_state.get("write_access_granted", False))
    if unlocked:
        return WriteAccessStatus(
            configured=True,
            unlocked=True,
            can_write=True,
            message="Guardado habilitado en esta sesión.",
        )

    return WriteAccessStatus(
        configured=True,
        unlocked=False,
        can_write=False,
        message="Ingresa la contraseña de escritura en la barra lateral para guardar resultados.",
    )


def render_write_access_panel(write_password: str | None, write_access_status: WriteAccessStatus) -> None:
    with st.sidebar:
        st.subheader("Acceso de escritura")
        st.caption("La app puede ser pública, pero el guardado en la base requiere esta contraseña.")

        if not write_password:
            st.warning(write_access_status.message)
            return

        if write_access_status.unlocked:
            st.success(write_access_status.message)
            if st.button("Bloquear guardado", use_container_width=True):
                st.session_state["write_access_granted"] = False
                st.rerun()
            return

        password_input = st.text_input(
            "Contraseña para guardar",
            type="password",
            key="storage_write_password_input",
        )
        if st.button("Habilitar guardado", use_container_width=True):
            if hmac.compare_digest(password_input, write_password):
                st.session_state["write_access_granted"] = True
                st.session_state["storage_write_password_input"] = ""
                st.rerun()

            st.session_state["write_access_granted"] = False
            st.error("Contraseña incorrecta. El guardado sigue bloqueado.")

        st.info(write_access_status.message)


def persist_text_results(
    storage_status: StorageStatus,
    write_access_status: WriteAccessStatus,
    source_text: str,
    results: list[dict[str, object]],
) -> None:
    if not storage_status.available:
        return
    if not write_access_status.can_write:
        st.info("Los resultados de texto no se guardaron. Habilita el guardado con la contraseña de escritura.")
        return

    try:
        save_text_evaluation(source_text, results)
        st.caption("La evaluación de texto se guardó en el historial.")
    except Exception as exc:
        st.warning("La evaluación de texto se calculó, pero no se pudo guardar en la base de datos.")
        st.caption(f"Detalle técnico: {exc}")


def persist_image_results(
    storage_status: StorageStatus,
    write_access_status: WriteAccessStatus,
    prompt_text: str,
    uploaded_files: list,
    fid_results: dict[str, float],
    clip_results: dict[str, float],
) -> None:
    if not storage_status.available:
        return
    if not write_access_status.can_write:
        st.info("Los resultados de imágenes no se guardaron. Habilita el guardado con la contraseña de escritura.")
        return

    try:
        save_image_evaluation(prompt_text, uploaded_files, fid_results, clip_results)
        st.caption("La evaluación de imágenes se guardó en el historial.")
    except Exception as exc:
        st.warning("La evaluación de imágenes se calculó, pero no se pudo guardar en la base de datos.")
        st.caption(f"Detalle técnico: {exc}")


def render_text_history(history: list[dict[str, object]]) -> None:
    st.subheader("Historial de texto")
    if not history:
        st.info("Todavía no hay evaluaciones de texto guardadas.")
        return

    for evaluation in history:
        created_at = format_timestamp(evaluation["created_at"])
        with st.expander(f"Evaluación de texto · {created_at}"):
            st.text_area(
                "Fuente",
                value=str(evaluation["source_text"]),
                height=140,
                disabled=True,
                key=f"text_source_{evaluation['id']}",
            )
            for candidate in evaluation["candidates"]:
                with st.container(border=True):
                    st.markdown(f"**{candidate['label']}**")
                    st.text_area(
                        candidate["label"],
                        value=str(candidate["candidate_text"]),
                        height=120,
                        disabled=True,
                        key=f"text_candidate_{evaluation['id']}_{candidate['slot']}",
                    )
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("BLEU", format_metric(float(candidate["bleu_score"])))
                    col2.metric("BERTScore Precision", format_metric(float(candidate["bert_precision"])))
                    col3.metric("BERTScore Recall", format_metric(float(candidate["bert_recall"])))
                    col4.metric("BERTScore F1", format_metric(float(candidate["bert_f1"])))


def render_image_history(history: list[dict[str, object]]) -> None:
    st.subheader("Historial de imágenes")
    if not history:
        st.info("Todavía no hay evaluaciones de imágenes guardadas.")
        return

    for evaluation in history:
        created_at = format_timestamp(evaluation["created_at"])
        with st.expander(f"Evaluación de imágenes · {created_at}"):
            st.text_area(
                "Texto de comparación",
                value=str(evaluation["prompt_text"]),
                height=100,
                disabled=True,
                key=f"image_prompt_{evaluation['id']}",
            )

            st.markdown("**Resultados FID**")
            fid_col1, fid_col2, fid_col3 = st.columns(3)
            fid_col1.metric("FID Imagen 1 vs Imágenes 2 y 3", format_metric(float(evaluation["fid_1_vs_23"])))
            fid_col2.metric("FID Imagen 2 vs Imágenes 1 y 3", format_metric(float(evaluation["fid_2_vs_13"])))
            fid_col3.metric("FID Imagen 3 vs Imágenes 1 y 2", format_metric(float(evaluation["fid_3_vs_12"])))

            st.markdown("**Resultados CLIPScore**")
            clip_col1, clip_col2, clip_col3 = st.columns(3)
            clip_col1.metric("CLIPScore Imagen 1 vs Texto", format_metric(float(evaluation["clip_1"])))
            clip_col2.metric("CLIPScore Imagen 2 vs Texto", format_metric(float(evaluation["clip_2"])))
            clip_col3.metric("CLIPScore Imagen 3 vs Texto", format_metric(float(evaluation["clip_3"])))

            image_columns = st.columns(3)
            for column, asset in zip(image_columns, evaluation["assets"]):
                with column:
                    st.markdown(f"**Imagen {asset['slot']}**")
                    st.caption(asset["filename"])
                    st.caption(f"SHA256: {asset['sha256']}")
                    try:
                        preview = Image.open(io.BytesIO(asset["image_bytes"]))
                        preview.load()
                        st.image(preview, use_container_width=True)
                    except Exception:
                        st.warning(f"No se pudo reconstruir la Imagen {asset['slot']} guardada.")


def render_history_tab(storage_status: StorageStatus) -> None:
    st.subheader("Historial")
    if not storage_status.available:
        st.info("El historial requiere una base PostgreSQL configurada mediante DATABASE_URL.")
        return

    st.caption(f"Mostrando las {HISTORY_LIMIT} evaluaciones más recientes de cada módulo.")

    try:
        text_history = list_recent_text_evaluations(limit=HISTORY_LIMIT)
        image_history = list_recent_image_evaluations(limit=HISTORY_LIMIT)
    except Exception as exc:
        st.error(f"No se pudo cargar el historial: {exc}")
        return

    render_text_history(text_history)
    st.divider()
    render_image_history(image_history)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")

    st.title(APP_TITLE)
    st.caption("Aplicación académica para evaluación multimodal con métricas de texto e imagen.")
    storage_status = get_storage_status()
    write_password = resolve_write_password()
    write_access_status = get_write_access_status(write_password)
    render_storage_banner(storage_status)
    render_write_access_panel(write_password, write_access_status)

    text_tab, image_tab, history_tab = st.tabs(["Evaluación de texto", "Evaluación de imágenes", "Historial"])

    with text_tab:
        st.subheader("Módulo de texto")
        with st.form("text_evaluation_form"):
            source_text = st.text_area(
                "Fuente",
                height=180,
                placeholder="Escribe aquí el texto fuente o referencia principal.",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                text_1 = st.text_area(
                    "Texto 1",
                    height=160,
                    placeholder="Ingresa el primer texto a comparar con la fuente.",
                )
            with col2:
                text_2 = st.text_area(
                    "Texto 2",
                    height=160,
                    placeholder="Ingresa el segundo texto a comparar con la fuente.",
                )
            with col3:
                text_3 = st.text_area(
                    "Texto 3",
                    height=160,
                    placeholder="Ingresa el tercer texto a comparar con la fuente.",
                )
            evaluate_text = st.form_submit_button("Evaluar textos", use_container_width=True)

        if evaluate_text:
            try:
                reference_text, candidates = validate_text_inputs(source_text, [text_1, text_2, text_3])

                with st.spinner("Procesando evaluación de texto..."):
                    text_results = []
                    for label, candidate in candidates:
                        bleu_score = compute_bleu(candidate, [reference_text])
                        bert_results = compute_bertscore(candidate, [reference_text])
                        text_results.append(
                            {
                                "label": label,
                                "candidate_text": candidate,
                                "bleu_score": bleu_score,
                                "bert_results": bert_results,
                            }
                        )

                render_text_results(text_results)
                persist_text_results(storage_status, write_access_status, reference_text, text_results)
            except Exception as exc:
                st.error(str(exc))

    with image_tab:
        st.subheader("Módulo de imágenes")
        st.warning(
            "FID con solo 3 imágenes es una aproximación exploratoria y no una evaluación "
            "estadísticamente robusta"
        )

        with st.form("image_evaluation_form"):
            image_text = st.text_input(
                "Texto para comparar con las imágenes",
                placeholder="Describe lo que deberían representar las imágenes.",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                image_1_file = st.file_uploader("Imagen 1", type=["png", "jpg", "jpeg"])
            with col2:
                image_2_file = st.file_uploader("Imagen 2", type=["png", "jpg", "jpeg"])
            with col3:
                image_3_file = st.file_uploader("Imagen 3", type=["png", "jpg", "jpeg"])
            evaluate_images = st.form_submit_button("Evaluar imágenes", use_container_width=True)

        if evaluate_images:
            uploaded_files = [image_1_file, image_2_file, image_3_file]
            try:
                cleaned_text = validate_image_inputs(image_text, uploaded_files)
                images = [load_image(file) for file in uploaded_files]

                with st.spinner("Procesando evaluación de imágenes..."):
                    fid_results = {
                        "fid_1_vs_23": compute_fid_single_vs_group(images[0], [images[1], images[2]]),
                        "fid_2_vs_13": compute_fid_single_vs_group(images[1], [images[0], images[2]]),
                        "fid_3_vs_12": compute_fid_single_vs_group(images[2], [images[0], images[1]]),
                    }
                    clip_results = {
                        "clip_1": compute_clipscore(images[0], cleaned_text),
                        "clip_2": compute_clipscore(images[1], cleaned_text),
                        "clip_3": compute_clipscore(images[2], cleaned_text),
                    }

                render_image_results(fid_results, clip_results)
                persist_image_results(
                    storage_status,
                    write_access_status,
                    cleaned_text,
                    uploaded_files,
                    fid_results,
                    clip_results,
                )
            except Exception as exc:
                st.error(str(exc))

    with history_tab:
        render_history_tab(storage_status)


if __name__ == "__main__":
    main()
