from __future__ import annotations

import io
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final
from urllib import error, request
from urllib.parse import urlparse

import streamlit as st
import torch
from bert_score import BERTScorer
from PIL import Image, UnidentifiedImageError
from sacrebleu.metrics import BLEU
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms import functional as TF

from storage import (
    MAX_IMAGE_SIZE_BYTES,
    StorageStatus,
    get_image_assets_for_evaluation,
    get_storage_status,
    list_recent_image_evaluations,
    list_recent_text_evaluations,
    resolve_database_url,
    save_image_evaluation,
    save_image_expert_review,
    save_text_evaluation,
    save_text_expert_review,
)


APP_TITLE: Final[str] = "Evaluador multimodal de texto e imágenes"
BERTSCORE_MODEL_NAME: Final[str] = "bert-base-multilingual-cased"
CLIP_MODEL_NAME: Final[str] = "openai/clip-vit-base-patch32"
FID_IMAGE_SIZE: Final[tuple[int, int]] = (299, 299)
DEVICE: Final[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_OPTIONS: Final[list[int]] = [1, 2, 3, 4, 5]
SCORE_CAPTIONS: Final[list[str]] = [
    "Muy deficiente",
    "Deficiente",
    "Aceptable",
    "Bueno",
    "Excelente",
]
SCORE_LABELS: Final[dict[int, str]] = dict(zip(SCORE_OPTIONS, SCORE_CAPTIONS))
HISTORY_PAGE_SIZE: Final[int] = 5
SUPABASE_SESSION_KEY: Final[str] = "supabase_auth_session"
SUPABASE_REFRESH_MARGIN_SECONDS: Final[int] = 60

TEXT_EXPERT_CRITERIA: Final[list[tuple[str, str, str]]] = [
    ("fidelidad_factual", "Fidelidad factual", "Conserva hechos, relaciones y atributos de la fuente sin errores."),
    ("conservacion_semantica", "Conservación semántica", "Mantiene el sentido general de la fuente aunque reformule."),
    ("rigor_historico", "Rigor histórico", "Usa un tratamiento compatible con contenido histórico y documental."),
    ("coherencia_claridad", "Coherencia y claridad", "El texto es comprensible, bien articulado y no se contradice."),
    ("ausencia_alucinaciones", "Ausencia de alucinaciones", "No introduce datos no sustentados por la fuente."),
    ("ausencia_anacronismos", "Ausencia de anacronismos", "Evita conceptos o enfoques impropios del contexto histórico."),
    ("adecuacion_comunicativa", "Adecuación comunicativa", "Resulta útil para fines académicos, educativos o divulgativos."),
]

IMAGE_EXPERT_CRITERIA: Final[list[tuple[str, str, str]]] = [
    ("correspondencia_texto", "Correspondencia con el texto", "La imagen refleja adecuadamente la descripción textual."),
    ("plausibilidad_historica", "Plausibilidad histórica", "La representación es verosímil para el contexto ecuatoriano."),
    ("coherencia_iconografica", "Coherencia iconográfica", "Los elementos visuales guardan relación entre sí."),
    ("adecuacion_contextual", "Adecuación contextual", "La escena o figura se ubica bien en su contexto cultural o histórico."),
    ("ausencia_anacronismos", "Ausencia de anacronismos visuales", "No aparecen objetos o estilos impropios del periodo tratado."),
    ("calidad_visual_funcional", "Calidad visual funcional", "La imagen es legible y útil para análisis o divulgación."),
    ("utilidad_divulgativa", "Utilidad educativa o divulgativa", "La imagen sería útil para explicar o comunicar el tema."),
]

TEXT_SUMMARY_QUESTIONS: Final[list[tuple[str, str]]] = [
    ("best_represents_source", "¿Cuál texto representa mejor la fuente histórica?"),
    ("clearest_for_dissemination", "¿Cuál texto es más claro para divulgación o docencia?"),
    ("highest_risk_error", "¿Cuál texto presenta mayor riesgo de error histórico?"),
]

IMAGE_SUMMARY_QUESTIONS: Final[list[tuple[str, str]]] = [
    ("best_historical_representation", "¿Cuál imagen representa mejor el contenido histórico solicitado?"),
    ("best_context_fit", "¿Cuál imagen tiene mejor adecuación visual al contexto ecuatoriano?"),
    ("highest_anachronism_risk", "¿Cuál imagen presenta mayor riesgo de anacronismo o error visual?"),
]


@dataclass(frozen=True)
class WriteAccessStatus:
    configured: bool
    unlocked: bool
    can_write: bool
    message: str


def clear_authenticated_session() -> None:
    st.session_state["app_authenticated"] = False
    st.session_state["write_access_granted"] = False
    st.session_state.pop(SUPABASE_SESSION_KEY, None)


def _get_secret_value(key: str) -> str | None:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        return None
    return None


def resolve_supabase_auth_config() -> tuple[str | None, str | None]:
    supabase_url = _get_secret_value("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    supabase_anon_key = _get_secret_value("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if supabase_url:
        supabase_url = supabase_url.rstrip("/")
    return supabase_url, supabase_anon_key


def extract_project_ref_from_supabase_url(supabase_url: str) -> str | None:
    parsed = urlparse(supabase_url)
    host = parsed.netloc.lower()
    if not host.endswith(".supabase.co"):
        return None
    project_ref = host.split(".")[0].strip()
    return project_ref or None


def extract_project_ref_from_database_url(database_url: str) -> str | None:
    parsed = urlparse(database_url)
    host = (parsed.hostname or "").lower()
    username = parsed.username or ""

    if host.startswith("db.") and host.endswith(".supabase.co"):
        parts = host.split(".")
        if len(parts) >= 3:
            return parts[1]

    if "." in username:
        candidate = username.split(".", 1)[1].strip().lower()
        if candidate:
            return candidate

    return None


def is_valid_supabase_project_url(supabase_url: str) -> bool:
    return supabase_url.startswith("https://") and ".supabase.co" in supabase_url


def build_supabase_auth_error_message(payload: dict[str, Any]) -> str:
    error_code = (
        str(payload.get("code") or "")
        or str(payload.get("error_code") or "")
        or str(payload.get("error") or "")
    ).strip()
    message = (
        str(payload.get("msg") or "")
        or str(payload.get("message") or "")
        or str(payload.get("error_description") or "")
        or str(payload.get("error") or "")
        or "Error de autenticación."
    ).strip()

    if error_code and error_code.lower() not in message.lower():
        return f"{message} [{error_code}]"
    return message


def has_supabase_auth_config() -> bool:
    supabase_url, supabase_anon_key = resolve_supabase_auth_config()
    return bool(supabase_url and supabase_anon_key)


def _supabase_auth_request(
    *,
    supabase_url: str,
    supabase_anon_key: str,
    path: str,
    method: str = "POST",
    payload: dict[str, Any] | None = None,
    access_token: str | None = None,
) -> dict[str, Any]:
    body = b""
    headers = {"apikey": supabase_anon_key}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    http_request = request.Request(
        f"{supabase_url}{path}",
        data=body if method.upper() != "GET" else None,
        headers=headers,
        method=method.upper(),
    )

    try:
        with request.urlopen(http_request, timeout=20) as response:
            raw_body = response.read().decode("utf-8")
            if not raw_body:
                return {}
            return dict(json.loads(raw_body))
    except error.HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            message = build_supabase_auth_error_message(dict(error_payload))
        except Exception:
            message = "Error de autenticación."

        if 400 <= exc.code < 500:
            raise ValueError(f"{message} (HTTP {exc.code})")
        raise RuntimeError(f"Supabase Auth no disponible (HTTP {exc.code}).") from exc
    except error.URLError as exc:
        reason = str(getattr(exc, "reason", "error de red"))
        raise RuntimeError(f"No se pudo conectar con Supabase Auth: {reason}.") from exc


def _store_supabase_session(auth_response: dict[str, Any]) -> None:
    access_token = str(auth_response.get("access_token") or "")
    refresh_token = str(auth_response.get("refresh_token") or "")
    expires_in = int(auth_response.get("expires_in") or 3600)
    user = auth_response.get("user") or {}
    user_email = str(user.get("email") or "")

    if not access_token or not refresh_token:
        raise RuntimeError("Supabase Auth no devolvió una sesión válida.")

    st.session_state[SUPABASE_SESSION_KEY] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time()) + max(30, expires_in),
        "user_email": user_email,
    }
    st.session_state["app_authenticated"] = True
    st.session_state["write_access_granted"] = True


def _refresh_supabase_session_if_needed(supabase_url: str, supabase_anon_key: str) -> bool:
    session = st.session_state.get(SUPABASE_SESSION_KEY)
    if not isinstance(session, dict):
        return False

    access_token = str(session.get("access_token") or "")
    refresh_token = str(session.get("refresh_token") or "")
    expires_at = int(session.get("expires_at") or 0)
    now = int(time.time())

    if access_token and expires_at - now > SUPABASE_REFRESH_MARGIN_SECONDS:
        st.session_state["app_authenticated"] = True
        st.session_state["write_access_granted"] = True
        return True

    if not refresh_token:
        clear_authenticated_session()
        return False

    try:
        refreshed = _supabase_auth_request(
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
            path="/auth/v1/token?grant_type=refresh_token",
            payload={"refresh_token": refresh_token},
        )
    except Exception:
        clear_authenticated_session()
        return False

    _store_supabase_session(refreshed)
    return True


def revoke_supabase_session() -> None:
    session = st.session_state.get(SUPABASE_SESSION_KEY) or {}
    access_token = str(session.get("access_token") or "")
    if not access_token:
        return

    supabase_url, supabase_anon_key = resolve_supabase_auth_config()
    if not supabase_url or not supabase_anon_key:
        return

    try:
        _supabase_auth_request(
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
            path="/auth/v1/logout",
            payload=None,
            access_token=access_token,
        )
    except Exception:
        # Logout revocation errors should not block local logout.
        pass


def require_app_authentication() -> None:
    """Block app rendering until the user authenticates with Supabase Auth."""
    supabase_url, supabase_anon_key = resolve_supabase_auth_config()
    if not supabase_url or not supabase_anon_key:
        st.subheader("Configuración de autenticación incompleta")
        st.error(
            "Define SUPABASE_URL y SUPABASE_ANON_KEY en secretos para habilitar login con Supabase."
        )
        st.stop()
    if not is_valid_supabase_project_url(supabase_url):
        st.subheader("SUPABASE_URL inválida")
        st.error(
            "SUPABASE_URL debe apuntar al proyecto, por ejemplo: "
            "`https://<project-ref>.supabase.co`."
        )
        st.caption(
            "No uses hosts de pooler (`aws-*.pooler.supabase.com`) para autenticación."
        )
        st.stop()

    database_url = resolve_database_url()
    auth_project_ref = extract_project_ref_from_supabase_url(supabase_url)
    db_project_ref = extract_project_ref_from_database_url(database_url) if database_url else None
    if auth_project_ref and db_project_ref and auth_project_ref != db_project_ref:
        st.subheader("Configuración inconsistente")
        st.error(
            "SUPABASE_URL y DATABASE_URL parecen pertenecer a proyectos distintos. "
            "Auth y DB deben apuntar al mismo project_ref."
        )
        st.caption(
            f"Auth project_ref detectado: `{auth_project_ref}` | "
            f"DB project_ref detectado: `{db_project_ref}`"
        )
        st.stop()

    if _refresh_supabase_session_if_needed(supabase_url, supabase_anon_key):
        return

    st.subheader("Iniciar sesión")
    st.caption("Autenticación gestionada por Supabase Auth (email y contraseña).")

    with st.form("supabase_login_form", clear_on_submit=True):
        user_email = st.text_input("Correo electrónico")
        user_password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Ingresar", use_container_width=True)

    if submitted:
        email_value = user_email.strip().lower()
        if not email_value or not user_password:
            st.error("Ingresa correo y contraseña.")
            st.stop()

        try:
            auth_response = _supabase_auth_request(
                supabase_url=supabase_url,
                supabase_anon_key=supabase_anon_key,
                path="/auth/v1/token?grant_type=password",
                payload={"email": email_value, "password": user_password},
            )
            _store_supabase_session(auth_response)
            st.rerun()
        except ValueError:
            clear_authenticated_session()
            st.error("No fue posible autenticarse con Supabase en este momento.")
        except Exception:
            clear_authenticated_session()
            st.error("No fue posible autenticarse con Supabase en este momento.")

    st.stop()


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


def image_to_fid_tensor(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor(resize_for_fid(image))


def prepare_fid_batch(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    tensors = [image_to_fid_tensor(image) for image in images]
    return torch.stack(tensors, dim=0).to(device)


def prepare_single_image_fid_approx_batch(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Approximate a one-image distribution with two deterministic views."""
    base_tensor = image_to_fid_tensor(image)
    augmented_tensor = torch.flip(base_tensor, dims=[2])

    if torch.equal(augmented_tensor, base_tensor):
        augmented_tensor = torch.roll(base_tensor, shifts=1, dims=2)

    if torch.equal(augmented_tensor, base_tensor):
        augmented_tensor = base_tensor.clone()
        augmented_tensor[0, 0, 0] = 255 - int(augmented_tensor[0, 0, 0].item())

    return torch.stack([base_tensor, augmented_tensor], dim=0).to(device)


def compute_fid_single_vs_group(target_image: Image.Image, reference_images: list[Image.Image]) -> float:
    """Compute FID for one image against a reference group."""
    if target_image is None:
        raise ValueError("La imagen objetivo no es válida.")
    if len(reference_images) < 2:
        raise ValueError("FID requiere al menos dos imágenes de referencia para este módulo.")

    try:
        metric = FrechetInceptionDistance(feature=64, normalize=False).to(DEVICE)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FID requiere dependencias adicionales. Verifica que 'torch-fidelity' esté instalado."
        ) from exc

    metric = metric.set_dtype(torch.float64)

    target_batch = prepare_single_image_fid_approx_batch(target_image, DEVICE)
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


@st.cache_resource(show_spinner=False)
def get_clip_components(device_name: str) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device_name)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor


def _coerce_clip_features(features: object) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and getattr(features, "pooler_output") is not None:
        return getattr(features, "pooler_output")
    if hasattr(features, "last_hidden_state") and getattr(features, "last_hidden_state") is not None:
        return getattr(features, "last_hidden_state")[:, 0, :]
    raise TypeError(f"No se pudieron extraer embeddings CLIP desde {type(features).__name__}.")


def compute_clipscore_compat(image: Image.Image, text: str) -> float:
    model, processor = get_clip_components(str(DEVICE))
    max_length = getattr(model.config.text_config, "max_position_embeddings", 77)

    image_inputs = processor(images=[image], return_tensors="pt", padding=True)
    text_inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    with torch.inference_mode():
        image_features = _coerce_clip_features(model.get_image_features(image_inputs["pixel_values"].to(DEVICE)))
        text_features = _coerce_clip_features(
            model.get_text_features(
                text_inputs["input_ids"].to(DEVICE),
                text_inputs["attention_mask"].to(DEVICE),
            )
        )

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return float((100 * (image_features * text_features).sum(dim=-1)).detach().cpu().item())


def compute_clipscore(image: Image.Image, text: str) -> float:
    """Compute CLIPScore between one image and one text."""
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Debes ingresar un texto para calcular CLIPScore.")

    metric = get_clip_metric(str(DEVICE))
    image_tensor = pil_to_tensor(image).to(DEVICE)

    with torch.inference_mode():
        try:
            metric.reset()
            clip_value = float(metric(image_tensor, cleaned_text).detach().cpu().item())
            metric.reset()
        except AttributeError as exc:
            metric.reset()
            if "norm" not in str(exc):
                raise
            clip_value = compute_clipscore_compat(image, cleaned_text)

    return clip_value


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "Sin fecha"
    return value.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def is_tab_selected(tab_container: Any) -> bool:
    """Return selected-state for modern tabs and stay compatible with older builds."""
    open_state = getattr(tab_container, "open", None)
    if open_state is None:
        return True
    return bool(open_state)


@st.cache_data(ttl=600, max_entries=100, show_spinner=False)
def get_cached_image_assets_for_history(evaluation_id: str) -> list[dict[str, object]]:
    return get_image_assets_for_evaluation(evaluation_id, include_bytes=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_text_history(limit: int, offset: int) -> list[dict[str, object]]:
    """Cache text history snapshots to avoid repeated DB reads on reruns."""
    return list_recent_text_evaluations(limit=limit, offset=offset)


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_image_history(limit: int, offset: int) -> list[dict[str, object]]:
    """Cache image history snapshots to avoid repeated DB reads on reruns."""
    return list_recent_image_evaluations(limit=limit, offset=offset)


def clear_history_caches() -> None:
    """Invalidate history caches after successful writes."""
    get_cached_text_history.clear()
    get_cached_image_history.clear()
    get_cached_image_assets_for_history.clear()


def pop_flash_message(key: str) -> str | None:
    message = st.session_state.get(key)
    if message:
        st.session_state.pop(key, None)
        return str(message)
    return None


def scale_markdown() -> str:
    return (
        "| Puntaje | Significado |\n"
        "|---|---|\n"
        "| 1 | Muy deficiente |\n"
        "| 2 | Deficiente |\n"
        "| 3 | Aceptable |\n"
        "| 4 | Bueno |\n"
        "| 5 | Excelente |"
    )


def render_safe_exception(exc: Exception, generic_message: str) -> None:
    """Show validation errors as-is and mask unexpected technical errors."""
    if isinstance(exc, ValueError):
        st.error(str(exc))
        return
    st.error(generic_message)


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


def validate_image_inputs(texts: list[str], uploaded_files: list) -> list[str]:
    if any(file is None for file in uploaded_files[:3]):
        raise ValueError("Debes cargar las imágenes 1, 2 y 3 antes de evaluar.")

    cleaned_texts: list[str] = []
    for index, text in enumerate(texts, start=1):
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError(f"Debes ingresar el texto para comparar con la Imagen {index}.")
        cleaned_texts.append(cleaned_text)

    size_limit_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        if uploaded_file is None:
            continue
        image_bytes = uploaded_file.getvalue()
        if not image_bytes:
            raise ValueError(f"La Imagen {index} no contiene datos.")
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"La Imagen {index} supera el límite de {size_limit_mb} MB.")
    return cleaned_texts


def validate_required_text(value: str, label: str) -> str:
    cleaned_value = value.strip()
    if not cleaned_value:
        raise ValueError(f"Debes completar el campo '{label}'.")
    return cleaned_value


def validate_score_value(value: int | None, label: str) -> int:
    if value is None:
        raise ValueError(f"Debes puntuar '{label}'.")
    return int(value)


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
    fid_col1.metric("FID Imagen 1 vs referencia", format_metric(fid_results["fid_1_vs_23"]))
    fid_col2.metric("FID Imagen 2 vs referencia", format_metric(fid_results["fid_2_vs_13"]))
    fid_col3.metric("FID Imagen 3 vs referencia", format_metric(fid_results["fid_3_vs_12"]))

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
    if not storage_status.available:
        st.warning(storage_status.message)


def get_write_access_status() -> WriteAccessStatus:
    if not has_supabase_auth_config():
        return WriteAccessStatus(
            configured=False,
            unlocked=False,
            can_write=False,
            message="Configura SUPABASE_URL y SUPABASE_ANON_KEY para habilitar autenticación y guardado.",
        )

    unlocked = bool(st.session_state.get("app_authenticated", False))
    if unlocked:
        session = st.session_state.get(SUPABASE_SESSION_KEY) or {}
        user_email = str(session.get("user_email") or "")
        label = f"Sesión autenticada ({user_email})." if user_email else "Sesión autenticada."
        return WriteAccessStatus(
            configured=True,
            unlocked=True,
            can_write=True,
            message=f"{label} Guardado habilitado en esta sesión.",
        )

    return WriteAccessStatus(
        configured=True,
        unlocked=False,
        can_write=False,
        message="Debes iniciar sesión con Supabase para usar la aplicación y guardar resultados.",
    )


def render_session_panel(write_access_status: WriteAccessStatus) -> None:
    with st.sidebar:
        st.subheader("Sesión")
        if write_access_status.unlocked:
            st.success(write_access_status.message)
            if st.button("Cerrar sesión", key="logout_session_button", use_container_width=True):
                revoke_supabase_session()
                clear_authenticated_session()
                st.rerun()
        else:
            st.warning(write_access_status.message)


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
        clear_history_caches()
        st.caption("La evaluación de texto se guardó en el historial.")
    except Exception as exc:
        st.warning("La evaluación de texto se calculó, pero no se pudo guardar en la base de datos.")
        st.caption(f"Tipo de error: {type(exc).__name__}")


def persist_image_results(
    storage_status: StorageStatus,
    write_access_status: WriteAccessStatus,
    prompt_texts: dict[int, str],
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
        save_image_evaluation(prompt_texts, uploaded_files, fid_results, clip_results)
        clear_history_caches()
        st.caption("La evaluación de imágenes se guardó en el historial.")
    except Exception as exc:
        st.warning("La evaluación de imágenes se calculó, pero no se pudo guardar en la base de datos.")
        st.caption(f"Tipo de error: {type(exc).__name__}")


def render_history_stepper(
    prefix: str,
    evaluations: list[dict[str, object]],
    start_index: int = 0,
) -> dict[str, object]:
    """Emulate a stepper with segmented control + prev/next for current Streamlit."""
    step_labels = [str(start_index + index + 1) for index in range(len(evaluations))]
    selector_key = f"{prefix}_history_stepper"

    if st.session_state.get(selector_key) not in step_labels:
        st.session_state[selector_key] = step_labels[0]

    current_label = str(st.session_state[selector_key])
    current_index = step_labels.index(current_label)

    prev_col, center_col, next_col = st.columns([1, 4, 1])
    with prev_col:
        if st.button("⬅️ Anterior", key=f"{selector_key}_prev", disabled=current_index == 0, use_container_width=True):
            st.session_state[selector_key] = step_labels[current_index - 1]
            st.rerun()
    with center_col:
        selected_label = st.segmented_control(
            "Stepper",
            step_labels,
            default=current_label,
            key=selector_key,
            label_visibility="collapsed",
            width="stretch",
        )
    with next_col:
        if st.button(
            "Siguiente ➡️",
            key=f"{selector_key}_next",
            disabled=current_index == len(step_labels) - 1,
            use_container_width=True,
        ):
            st.session_state[selector_key] = step_labels[current_index + 1]
            st.rerun()

    resolved_label = str(selected_label or step_labels[0])
    resolved_index = step_labels.index(resolved_label)
    selected_evaluation = evaluations[resolved_index]
    st.caption(
        f"Registro {start_index + resolved_index + 1} · "
        f"{format_timestamp(selected_evaluation['created_at'])}"
    )
    return selected_evaluation


def render_history_page_controls(prefix: str, page_size: int = HISTORY_PAGE_SIZE) -> tuple[int, int, int]:
    """Render simple previous/next pagination controls backed by session state."""
    offset_key = f"{prefix}_history_offset"
    if offset_key not in st.session_state:
        st.session_state[offset_key] = 0

    offset = max(0, int(st.session_state[offset_key]))
    current_page = (offset // page_size) + 1

    prev_col, center_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        if st.button(
            "⬅️ Página anterior",
            key=f"{offset_key}_prev",
            disabled=offset == 0,
            use_container_width=True,
        ):
            st.session_state[offset_key] = max(0, offset - page_size)
            st.rerun()
    with center_col:
        st.markdown(
            (
                "<p style='text-align: center; font-weight: 700; margin: 0.35rem 0 0;'>"
                "Página <span style='color: #ff6b57;'>"
                f"{current_page}"
                "</span></p>"
            ),
            unsafe_allow_html=True,
        )
    with next_col:
        if st.button(
            "Página siguiente ➡️",
            key=f"{offset_key}_next",
            use_container_width=True,
        ):
            st.session_state[offset_key] = offset + page_size
            st.rerun()

    return offset, page_size, current_page


def render_text_history_snapshot(evaluation: dict[str, object]) -> None:
    st.text_area(
        "Fuente",
        value=str(evaluation["source_text"]),
        height=360,
        disabled=True,
        key=f"text_source_snapshot_{evaluation['id']}",
    )
    for candidate in evaluation["candidates"]:
        with st.container(border=True):
            st.markdown(f"**{candidate['label']}**")
            st.text_area(
                candidate["label"],
                value=str(candidate["candidate_text"]),
                height=360,
                disabled=True,
                key=f"text_candidate_snapshot_{evaluation['id']}_{candidate['slot']}",
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("BLEU", format_metric(float(candidate["bleu_score"])))
            col2.metric("BERTScore Precision", format_metric(float(candidate["bert_precision"])))
            col3.metric("BERTScore Recall", format_metric(float(candidate["bert_recall"])))
            col4.metric("BERTScore F1", format_metric(float(candidate["bert_f1"])))


def render_image_history_snapshot(evaluation: dict[str, object]) -> None:
    prompt_texts = evaluation.get("prompt_texts", {})
    prompt_columns = st.columns(3)
    for slot in range(1, 4):
        with prompt_columns[slot - 1]:
            st.text_area(
                f"Texto Imagen {slot}",
                value=str(prompt_texts.get(slot, "")),
                height=120,
                disabled=True,
                key=f"image_prompt_snapshot_{evaluation['id']}_{slot}",
            )

    st.markdown("**Resultados FID**")
    fid_col1, fid_col2, fid_col3 = st.columns(3)
    fid_col1.metric("FID Imagen 1 vs referencia", format_metric(float(evaluation["fid_1_vs_23"])))
    fid_col2.metric("FID Imagen 2 vs referencia", format_metric(float(evaluation["fid_2_vs_13"])))
    fid_col3.metric("FID Imagen 3 vs referencia", format_metric(float(evaluation["fid_3_vs_12"])))

    st.markdown("**Resultados CLIPScore**")
    clip_col1, clip_col2, clip_col3 = st.columns(3)
    clip_col1.metric("CLIPScore Imagen 1 vs Texto", format_metric(float(evaluation["clip_1"])))
    clip_col2.metric("CLIPScore Imagen 2 vs Texto", format_metric(float(evaluation["clip_2"])))
    clip_col3.metric("CLIPScore Imagen 3 vs Texto", format_metric(float(evaluation["clip_3"])))

    st.markdown("**Activos guardados**")
    image_columns = st.columns(4)
    for column, asset in zip(image_columns, evaluation["assets"]):
        with column:
            st.markdown(f"**Imagen {asset['slot']}**")
            st.caption(asset["filename"])
            st.caption(f"SHA256: {asset['sha256']}")

        preview_toggle_key = f"load_previews_{evaluation['id']}"
    load_previews = st.toggle(
        "Cargar previsualizaciones guardadas",
        key=preview_toggle_key,
        value=True,
    )

    if not load_previews:
        return

    try:
        preview_assets = get_cached_image_assets_for_history(str(evaluation["id"]))
    except Exception as exc:
        st.warning(f"No se pudieron cargar las imágenes guardadas: {exc}")
        return

    preview_columns = st.columns(4)
    for column, asset in zip(preview_columns, preview_assets):
        with column:
            st.markdown(f"**Preview Imagen {asset['slot']}**")
            try:
                preview = Image.open(io.BytesIO(asset["image_bytes"]))
                preview.load()
                st.image(preview, use_container_width=True)
            except Exception:
                st.warning(f"No se pudo reconstruir la Imagen {asset['slot']} guardada.")


def render_expert_evaluation_header(form_prefix: str) -> tuple[str, str, str]:
    with st.container(border=True):
        st.markdown("#### Instrucciones para la evaluación experta")
        info_col, evaluator_col = st.columns([1.3, 1.0])
        with info_col:
            st.markdown(
                "1. Revisa la evaluación seleccionada.\n"
                "2. Puntúa cada criterio usando la escala de 1 a 5.\n"
                "3. Completa las preguntas comparativas.\n"
                "4. Añade observaciones solo si hace falta una nota cualitativa adicional."
            )
            st.markdown(scale_markdown())
        with evaluator_col:
            evaluator_name = st.text_input(
                "Nombre de la persona evaluadora",
                key=f"{form_prefix}_evaluator_name",
                placeholder="Nombre completo",
            )
            evaluator_specialty = st.text_input(
                "Especialidad",
                key=f"{form_prefix}_evaluator_specialty",
                placeholder="Historia, archivo, docencia, etc.",
            )
            evaluator_institution = st.text_input(
                "Institución",
                key=f"{form_prefix}_evaluator_institution",
                placeholder="Universidad, archivo, museo, etc.",
            )
    return evaluator_name, evaluator_specialty, evaluator_institution


def render_score_input(label: str, key: str, help_text: str) -> int | None:
    st.markdown(f"**{label}**")
    st.caption(help_text)
    return st.segmented_control(
        f"Selecciona un puntaje para {label.lower()}",
        SCORE_OPTIONS,
        default=None,
        required=True,
        key=key,
        width="stretch",
        label_visibility="collapsed",
    )


def render_text_expert_review_form(
    evaluation: dict[str, object],
    storage_status: StorageStatus,
    write_access_status: WriteAccessStatus,
) -> None:
    st.markdown("### Nueva evaluación experta de texto")
    if not write_access_status.can_write:
        st.info("Habilita el guardado con la contraseña de escritura para registrar una evaluación experta.")
        return

    form_prefix = f"text_expert_{evaluation['id']}"
    candidate_labels = [str(candidate["label"]) for candidate in evaluation["candidates"]]

    with st.form(f"{form_prefix}_form", clear_on_submit=True):
        evaluator_name, evaluator_specialty, evaluator_institution = render_expert_evaluation_header(form_prefix)

        candidate_responses: dict[int, dict[str, int | None]] = {}
        for candidate in evaluation["candidates"]:
            slot = int(candidate["slot"])
            candidate_responses[slot] = {}
            with st.container(border=True):
                st.markdown(f"#### {candidate['label']}")
                for criterion_key, criterion_label, criterion_help in TEXT_EXPERT_CRITERIA:
                    candidate_responses[slot][criterion_key] = render_score_input(
                        criterion_label,
                        key=f"{form_prefix}_candidate_{slot}_{criterion_key}",
                        help_text=criterion_help,
                    )

        st.markdown("#### Comparación global")
        summary_answers: dict[str, str | None] = {}
        for question_key, question_label in TEXT_SUMMARY_QUESTIONS:
            summary_answers[question_key] = st.selectbox(
                question_label,
                options=candidate_labels,
                index=None,
                placeholder="Selecciona una opción",
                key=f"{form_prefix}_{question_key}",
            )

        observations = st.text_area(
            "Observaciones (opcional)",
            height=140,
            key=f"{form_prefix}_observations",
            placeholder="Añade aquí observaciones cualitativas si lo consideras necesario.",
        )
        submitted = st.form_submit_button("Enviar evaluación experta de texto", use_container_width=True)

    if not submitted:
        return

    try:
        cleaned_name = validate_required_text(evaluator_name, "Nombre de la persona evaluadora")
        cleaned_specialty = validate_required_text(evaluator_specialty, "Especialidad")
        cleaned_institution = validate_required_text(evaluator_institution, "Institución")

        validated_candidates: dict[str, dict[str, object]] = {}
        for candidate in evaluation["candidates"]:
            slot = int(candidate["slot"])
            scores: dict[str, int] = {}
            for criterion_key, criterion_label, _ in TEXT_EXPERT_CRITERIA:
                scores[criterion_key] = validate_score_value(
                    candidate_responses[slot][criterion_key],
                    f"{candidate['label']} · {criterion_label}",
                )
            validated_candidates[str(slot)] = {
                "label": str(candidate["label"]),
                "scores": scores,
            }

        validated_summary = {
            question_key: validate_required_text(str(summary_answers[question_key] or ""), question_label)
            for question_key, question_label in TEXT_SUMMARY_QUESTIONS
        }

        responses = {
            "kind": "text",
            "scale": SCORE_LABELS,
            "candidates": validated_candidates,
            "summary": validated_summary,
        }
        cleaned_observations = observations.strip() or None
        save_text_expert_review(
            evaluation["id"],
            cleaned_name,
            cleaned_specialty,
            cleaned_institution,
            responses,
            observations=cleaned_observations,
        )
        clear_history_caches()
        st.session_state["text_expert_review_flash"] = "La evaluación experta de texto se guardó correctamente."
        st.rerun()
    except Exception as exc:
        render_safe_exception(exc, "No se pudo guardar la evaluación experta de texto.")


def render_image_expert_review_form(
    evaluation: dict[str, object],
    storage_status: StorageStatus,
    write_access_status: WriteAccessStatus,
) -> None:
    st.markdown("### Nueva evaluación experta de imágenes")
    if not write_access_status.can_write:
        st.info("Habilita el guardado con la contraseña de escritura para registrar una evaluación experta.")
        return

    form_prefix = f"image_expert_{evaluation['id']}"
    image_labels = [f"Imagen {slot}" for slot in range(1, 4)]
    has_reference_image = any(int(asset["slot"]) == 4 for asset in evaluation["assets"])

    with st.form(f"{form_prefix}_form", clear_on_submit=True):
        evaluator_name, evaluator_specialty, evaluator_institution = render_expert_evaluation_header(form_prefix)

        image_responses: dict[int, dict[str, int | None]] = {}
        for slot in range(1, 4):
            image_responses[slot] = {}
            with st.container(border=True):
                st.markdown(f"#### Imagen {slot}")
                for criterion_key, criterion_label, criterion_help in IMAGE_EXPERT_CRITERIA:
                    image_responses[slot][criterion_key] = render_score_input(
                        criterion_label,
                        key=f"{form_prefix}_image_{slot}_{criterion_key}",
                        help_text=criterion_help,
                    )

        reference_score: int | None = None
        if has_reference_image:
            st.markdown("#### Imagen 4 de referencia")
            reference_score = render_score_input(
                "Pertinencia de la Imagen 4 como referencia adicional",
                key=f"{form_prefix}_reference_score",
                help_text="Valora si la cuarta imagen aportó valor como referencia comparativa.",
            )

        st.markdown("#### Comparación global")
        summary_answers: dict[str, str | None] = {}
        for question_key, question_label in IMAGE_SUMMARY_QUESTIONS:
            summary_answers[question_key] = st.selectbox(
                question_label,
                options=image_labels,
                index=None,
                placeholder="Selecciona una opción",
                key=f"{form_prefix}_{question_key}",
            )

        observations = st.text_area(
            "Observaciones (opcional)",
            height=140,
            key=f"{form_prefix}_observations",
            placeholder="Añade aquí observaciones cualitativas si lo consideras necesario.",
        )
        submitted = st.form_submit_button("Enviar evaluación experta de imágenes", use_container_width=True)

    if not submitted:
        return

    try:
        cleaned_name = validate_required_text(evaluator_name, "Nombre de la persona evaluadora")
        cleaned_specialty = validate_required_text(evaluator_specialty, "Especialidad")
        cleaned_institution = validate_required_text(evaluator_institution, "Institución")

        validated_images: dict[str, dict[str, object]] = {}
        for slot in range(1, 4):
            scores: dict[str, int] = {}
            for criterion_key, criterion_label, _ in IMAGE_EXPERT_CRITERIA:
                scores[criterion_key] = validate_score_value(
                    image_responses[slot][criterion_key],
                    f"Imagen {slot} · {criterion_label}",
                )
            validated_images[str(slot)] = {
                "label": f"Imagen {slot}",
                "scores": scores,
            }

        validated_summary = {
            question_key: validate_required_text(str(summary_answers[question_key] or ""), question_label)
            for question_key, question_label in IMAGE_SUMMARY_QUESTIONS
        }

        responses = {
            "kind": "image",
            "scale": SCORE_LABELS,
            "images": validated_images,
            "summary": validated_summary,
            "reference_image": {
                "present": has_reference_image,
                "score": (
                    validate_score_value(reference_score, "Pertinencia de la Imagen 4 como referencia adicional")
                    if has_reference_image
                    else None
                ),
            },
        }
        cleaned_observations = observations.strip() or None
        save_image_expert_review(
            evaluation["id"],
            cleaned_name,
            cleaned_specialty,
            cleaned_institution,
            responses,
            observations=cleaned_observations,
        )
        clear_history_caches()
        st.session_state["image_expert_review_flash"] = "La evaluación experta de imágenes se guardó correctamente."
        st.rerun()
    except Exception as exc:
        render_safe_exception(exc, "No se pudo guardar la evaluación experta de imágenes.")


def render_scored_review_block(
    title: str,
    values: dict[str, int],
    criteria: list[tuple[str, str, str]],
) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        lines = []
        for criterion_key, criterion_label, _ in criteria:
            score = values.get(criterion_key)
            if score is None:
                continue
            lines.append(f"- {criterion_label}: **{score}/5** ({SCORE_LABELS.get(int(score), 'Sin escala')})")
        st.markdown("\n".join(lines) if lines else "_Sin respuestas registradas._")


def render_text_expert_reviews(reviews: list[dict[str, object]]) -> None:
    st.markdown("### Evaluaciones expertas guardadas")
    if not reviews:
        st.info("Todavía no hay evaluaciones expertas guardadas para esta evaluación de texto.")
        return

    for review in reviews:
        header = f"{review['display_name']} · {format_timestamp(review['created_at'])}"
        with st.expander(header):
            st.caption(f"{review['evaluator_specialty']} · {review['evaluator_institution']}")
            responses = review.get("responses", {})
            for candidate in responses.get("candidates", {}).values():
                render_scored_review_block(
                    str(candidate.get("label", "Texto")),
                    dict(candidate.get("scores", {})),
                    TEXT_EXPERT_CRITERIA,
                )

            summary = responses.get("summary", {})
            if summary:
                st.markdown("**Comparación global**")
                st.markdown(
                    "\n".join(
                        f"- {question_label}: **{summary.get(question_key, 'Sin respuesta')}**"
                        for question_key, question_label in TEXT_SUMMARY_QUESTIONS
                    )
                )

            if review.get("observations"):
                st.markdown("**Observaciones**")
                st.write(str(review["observations"]))


def render_image_expert_reviews(reviews: list[dict[str, object]]) -> None:
    st.markdown("### Evaluaciones expertas guardadas")
    if not reviews:
        st.info("Todavía no hay evaluaciones expertas guardadas para esta evaluación de imágenes.")
        return

    for review in reviews:
        header = f"{review['display_name']} · {format_timestamp(review['created_at'])}"
        with st.expander(header):
            st.caption(f"{review['evaluator_specialty']} · {review['evaluator_institution']}")
            responses = review.get("responses", {})
            for image in responses.get("images", {}).values():
                render_scored_review_block(
                    str(image.get("label", "Imagen")),
                    dict(image.get("scores", {})),
                    IMAGE_EXPERT_CRITERIA,
                )

            summary = responses.get("summary", {})
            if summary:
                st.markdown("**Comparación global**")
                st.markdown(
                    "\n".join(
                        f"- {question_label}: **{summary.get(question_key, 'Sin respuesta')}**"
                        for question_key, question_label in IMAGE_SUMMARY_QUESTIONS
                    )
                )

            reference_info = responses.get("reference_image", {})
            if reference_info.get("present"):
                reference_score = reference_info.get("score")
                st.markdown(
                    f"**Imagen 4 de referencia**: {reference_score}/5 "
                    f"({SCORE_LABELS.get(int(reference_score), 'Sin escala')})"
                )

            if review.get("observations"):
                st.markdown("**Observaciones**")
                st.write(str(review["observations"]))


def render_text_history_section(
    history: list[dict[str, object]],
    storage_status: StorageStatus,
    write_access_status: WriteAccessStatus,
    offset: int = 0,
) -> None:
    st.subheader("Historial de textos turisticos")
    flash_message = pop_flash_message("text_expert_review_flash")
    if flash_message:
        st.success(flash_message)
    if not history:
        st.info("Todavía no hay evaluaciones de texto guardadas.")
        return

    selected_evaluation = render_history_stepper("text", history, start_index=offset)
    render_text_history_snapshot(selected_evaluation)
    st.divider()
    render_text_expert_reviews(list(selected_evaluation.get("expert_reviews", [])))
    st.divider()
    render_text_expert_review_form(selected_evaluation, storage_status, write_access_status)


def render_image_history_section(
    history: list[dict[str, object]],
    storage_status: StorageStatus,
    write_access_status: WriteAccessStatus,
    offset: int = 0,
) -> None:
    st.subheader("Historial de imágenes")
    flash_message = pop_flash_message("image_expert_review_flash")
    if flash_message:
        st.success(flash_message)
    if not history:
        st.info("Todavía no hay evaluaciones de imágenes guardadas.")
        return

    selected_evaluation = render_history_stepper("image", history, start_index=offset)
    render_image_history_snapshot(selected_evaluation)
    st.divider()
    render_image_expert_reviews(list(selected_evaluation.get("expert_reviews", [])))
    st.divider()
    render_image_expert_review_form(selected_evaluation, storage_status, write_access_status)


def render_history_tab(storage_status: StorageStatus, write_access_status: WriteAccessStatus) -> None:
    st.subheader("Historial")
    if not storage_status.available:
        st.info("El historial requiere una base PostgreSQL configurada mediante DATABASE_URL.")
        return

    st.caption("El historial se muestra por páginas.")

    image_history_tab, text_history_tab = st.tabs(
        ["Imágenes", "Texto"],
        key="history_sections",
        on_change="rerun",
    )

    with image_history_tab:
        if is_tab_selected(image_history_tab):
            image_offset, image_limit, _image_page = render_history_page_controls("image")
            try:
                image_history = get_cached_image_history(limit=image_limit, offset=image_offset)
            except Exception as exc:
                st.error(f"No se pudo cargar el historial de imágenes ({type(exc).__name__}).")
                return
            render_image_history_section(image_history, storage_status, write_access_status, offset=image_offset)

    with text_history_tab:
        if is_tab_selected(text_history_tab):
            text_offset, text_limit, _text_page = render_history_page_controls("text")
            try:
                text_history = get_cached_text_history(limit=text_limit, offset=text_offset)
            except Exception as exc:
                st.error(f"No se pudo cargar el historial de texto ({type(exc).__name__}).")
                return
            render_text_history_section(text_history, storage_status, write_access_status, offset=text_offset)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")

    st.title(APP_TITLE)
    st.caption("Aplicación académica para evaluación multimodal con métricas de texto e imagen.")
    require_app_authentication()
    write_access_status = get_write_access_status()
    render_session_panel(write_access_status)
    storage_status = get_storage_status()
    render_storage_banner(storage_status)

    history_tab, text_tab, image_tab = st.tabs(
        ["Historial", "Evaluación de texto", "Evaluación de imágenes"],
        key="main_sections",
        on_change="rerun",
    )

    with history_tab:
        render_history_tab(storage_status, write_access_status)

    with text_tab:
        st.subheader("Módulo de texto")
        with st.form("text_evaluation_form"):
            source_text = st.text_area(
                "Fuente",
                height=360,
                placeholder="Escribe aquí el texto fuente o referencia principal.",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                text_1 = st.text_area(
                    "Texto 1",
                    height=360,
                    placeholder="Ingresa el primer texto a comparar con la fuente.",
                )
            with col2:
                text_2 = st.text_area(
                    "Texto 2",
                    height=360,
                    placeholder="Ingresa el segundo texto a comparar con la fuente.",
                )
            with col3:
                text_3 = st.text_area(
                    "Texto 3",
                    height=360,
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
                render_safe_exception(exc, "No se pudo completar la evaluación de texto.")

    with image_tab:
        st.subheader("Módulo de imágenes")
        st.caption(
            "La Imagen 4 es opcional y se usa como referencia adicional en FID. "
            "Para evitar fallos numéricos con un solo target, la app aproxima la distribución "
            "de cada imagen objetivo con una segunda vista determinística."
        )

        with st.form("image_evaluation_form"):
            text_col1, text_col2, text_col3 = st.columns(3)
            with text_col1:
                image_text_1 = st.text_area(
                    "Texto para Imagen 1",
                    height=360,
                    placeholder="Describe lo que debería representar la Imagen 1.",
                )
            with text_col2:
                image_text_2 = st.text_area(
                    "Texto para Imagen 2",
                    height=360,
                    placeholder="Describe lo que debería representar la Imagen 2.",
                )
            with text_col3:
                image_text_3 = st.text_area(
                    "Texto para Imagen 3",
                    height=360,
                    placeholder="Describe lo que debería representar la Imagen 3.",
                )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                image_1_file = st.file_uploader("Imagen 1", type=["png", "jpg", "jpeg"])
            with col2:
                image_2_file = st.file_uploader("Imagen 2", type=["png", "jpg", "jpeg"])
            with col3:
                image_3_file = st.file_uploader("Imagen 3", type=["png", "jpg", "jpeg"])
            with col4:
                image_4_file = st.file_uploader("Imagen 4 (referencia opcional)", type=["png", "jpg", "jpeg"])
            evaluate_images = st.form_submit_button("Evaluar imágenes", use_container_width=True)

        if evaluate_images:
            uploaded_files = [image_1_file, image_2_file, image_3_file, image_4_file]
            try:
                cleaned_texts = validate_image_inputs(
                    [image_text_1, image_text_2, image_text_3],
                    uploaded_files,
                )
                images = [load_image(file) if file is not None else None for file in uploaded_files]
                reference_sets = {
                    "fid_1_vs_23": [images[1], images[2]] + ([images[3]] if images[3] is not None else []),
                    "fid_2_vs_13": [images[0], images[2]] + ([images[3]] if images[3] is not None else []),
                    "fid_3_vs_12": [images[0], images[1]] + ([images[3]] if images[3] is not None else []),
                }

                with st.spinner("Procesando evaluación de imágenes..."):
                    fid_results = {
                        "fid_1_vs_23": compute_fid_single_vs_group(images[0], reference_sets["fid_1_vs_23"]),
                        "fid_2_vs_13": compute_fid_single_vs_group(images[1], reference_sets["fid_2_vs_13"]),
                        "fid_3_vs_12": compute_fid_single_vs_group(images[2], reference_sets["fid_3_vs_12"]),
                    }
                    clip_results = {
                        "clip_1": compute_clipscore(images[0], cleaned_texts[0]),
                        "clip_2": compute_clipscore(images[1], cleaned_texts[1]),
                        "clip_3": compute_clipscore(images[2], cleaned_texts[2]),
                    }

                render_image_results(fid_results, clip_results)
                persist_image_results(
                    storage_status,
                    write_access_status,
                    {1: cleaned_texts[0], 2: cleaned_texts[1], 3: cleaned_texts[2]},
                    uploaded_files,
                    fid_results,
                    clip_results,
                )
            except Exception as exc:
                render_safe_exception(exc, "No se pudo completar la evaluación de imágenes.")



if __name__ == "__main__":
    main()
