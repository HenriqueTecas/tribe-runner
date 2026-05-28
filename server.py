"""TRIBE v2 WebSocket backend.

Loads the model once at startup, then each WebSocket message is a phrase;
each response is a JSON header (phrase, top regions, signal stats) followed
by a GLB binary frame (sulcal-shaded brain with thresholded transparency).

Run on the pod:
    pip install fastapi 'uvicorn[standard]' websockets
    python /workspace/tribev2/server.py

By default binds 0.0.0.0:8000. Tunnel from your laptop with:
    ssh -p <port> -i <key> -L 8000:localhost:8000 root@<pod-ip>
"""

import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pyvista as pv
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import tribev2.main as _tribe_main

# TRIBE's get_loaders() calls _free_extractor_model() after each prepare() to
# free VRAM. That makes a hot-reload server reload LLaMA / V-JEPA2 / Wav2Vec-BERT
# weights on every request (~10–60s each). On a 24GB 3090 we have headroom to
# keep them resident; no-op the freeing so subsequent calls reuse the warm model.
_tribe_main._free_extractor_model = lambda *_a, **_kw: None

from tribev2 import TribeModel
from tribev2.plotting import PlotBrainPyvista
from tribev2.plotting.utils import get_cmap, get_scalar_mappable
from tribev2.utils import get_hcp_labels, summarize_by_roi

pv.OFF_SCREEN = True

CACHE = os.environ.get("TRIBE_CACHE", "/workspace/.cache/tribev2")
WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))
HOST = os.environ.get("TRIBE_HOST", "0.0.0.0")
PORT = int(os.environ.get("TRIBE_PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tribe-server")

state: dict = {}


# ---------------------------------------------------------------------------
# Region descriptions (HCP-MMP1 / Glasser 2016)
# ---------------------------------------------------------------------------

# 22 macro-cluster groupings from Glasser 2016 supplementary. Used as a
# fallback description for any fine-region that isn't in REGION_DESCRIPTIONS.
MACRO_DESCRIPTIONS: dict[str, str] = {
    "Primary_Visual_Cortex": "Primary visual cortex (V1) — first cortical stage of vision; retinotopic.",
    "Early_Visual_Cortex": "Early visual cortex (V2/V3/V4) — orientation, color, low-level form processing.",
    "Dorsal_Stream_Visual_Cortex": "Dorsal-stream visual cortex — 'where/how' pathway: spatial layout and motion for action.",
    "Ventral_Stream_Visual_Cortex": "Ventral-stream visual cortex — 'what' pathway: object, face and scene recognition.",
    "MT+_Complex_and_Neighboring_Visual_Areas": "MT+ complex — visual motion processing and motion-defined form.",
    "Somatosensory_and_Motor_Cortex": "Somatosensory and primary motor cortex — touch perception and voluntary movement.",
    "Paracentral_Lobular_and_Mid_Cingulate_Cortex": "Paracentral / mid-cingulate — leg/foot somatomotor and cognitive-motor control.",
    "Premotor_Cortex": "Premotor cortex — motor planning, action selection, sequence execution.",
    "Posterior_Opercular_Cortex": "Posterior opercular cortex — orofacial somatosensation, swallowing, speech motor.",
    "Early_Auditory_Cortex": "Early auditory cortex (A1, belt) — basic frequency and spectrotemporal processing.",
    "Auditory_Association_Cortex": "Auditory association cortex — speech, music, complex sound recognition.",
    "Insular_and_Frontal_Opercular_Cortex": "Insular / frontal opercular cortex — interoception, salience, taste, pain affect.",
    "Medial_Temporal_Cortex": "Medial temporal cortex — declarative memory, scene/place recognition (parahippocampal).",
    "Lateral_Temporal_Cortex": "Lateral temporal cortex — semantic memory, lexical access, social/biological perception.",
    "Temporo-Parieto-Occipital_Junction": "Temporo-parieto-occipital junction — multimodal integration, theory-of-mind, language.",
    "Superior_Parietal_Cortex": "Superior parietal cortex — visuospatial attention, reaching, body-centered coordinates.",
    "Inferior_Parietal_Cortex": "Inferior parietal cortex — attention, language, tool use, mathematical cognition.",
    "Posterior_Cingulate_Cortex": "Posterior cingulate / precuneus — default-mode hub, self-referential thought, autobiographical memory.",
    "Anterior_Cingulate_and_Medial_Prefrontal_Cortex": "Anterior cingulate / medial PFC — conflict monitoring, value, social cognition, emotion regulation.",
    "Orbital_and_Polar_Frontal_Cortex": "Orbital / polar frontal cortex — reward valuation, decision-making, future planning.",
    "Inferior_Frontal_Cortex": "Inferior frontal cortex — speech production (Broca, left), inhibitory control (right).",
    "DorsoLateral_Prefrontal_Cortex": "Dorsolateral prefrontal cortex — working memory, executive control, goal maintenance.",
}

# Curated specific descriptions for well-known HCP-MMP fine regions. Anything
# missing falls back to MACRO_DESCRIPTIONS via fine→macro mapping built at
# startup. Region names are bare (no -lh/-rh suffix) — they're the same on
# both hemispheres in the parcellation.
REGION_DESCRIPTIONS: dict[str, str] = {
    "V1": "Primary visual cortex — first cortical processing stage; retinotopic edge & orientation.",
    "V2": "Secondary visual cortex — contour, illusory shape, figure/ground.",
    "V3": "Visual area V3 — form and motion integration in dorsal stream.",
    "V3A": "V3A — coherent global motion and 3D structure-from-motion.",
    "V3B": "V3B — disparity and stereoscopic depth.",
    "V3CD": "V3CD — caudal V3 division, motion-related.",
    "V4": "V4 — color, hue, and intermediate form features.",
    "V4t": "V4 transitional zone — motion/form interface near MT+.",
    "V6": "V6 — egomotion and visual self-motion processing.",
    "V6A": "V6A — visually-guided reaching and grasping.",
    "V7": "V7 — dorsal extrastriate cortex; spatial attention.",
    "V8": "V8 — color-selective ventral region adjacent to V4.",
    "MT": "Middle Temporal area (V5) — visual motion direction and speed.",
    "MST": "Medial Superior Temporal — optic flow, self-motion estimation.",
    "FST": "FST — motion processing in posterior STS region.",
    "LO1": "Lateral Occipital 1 — object shape; part of object recognition stream.",
    "LO2": "Lateral Occipital 2 — object shape; part of object recognition stream.",
    "LO3": "Lateral Occipital 3 — object shape and category processing.",
    "PIT": "Posterior Inferotemporal — object category recognition.",
    "FFC": "Fusiform Face Complex — face identity (overlaps the canonical FFA).",
    "VVC": "Ventral Visual Complex — high-level object/word/face perception (includes VWFA).",
    "TF": "TF — parahippocampal/scene recognition (overlaps PPA).",
    "PHA1": "Parahippocampal Area 1 — scene perception and spatial layout.",
    "PHA2": "Parahippocampal Area 2 — scene perception and spatial layout.",
    "PHA3": "Parahippocampal Area 3 — scene perception and spatial layout.",
    "VMV1": "Ventromedial Visual 1 — peripheral visual field; medial ventral stream.",
    "VMV2": "Ventromedial Visual 2 — peripheral visual field; medial ventral stream.",
    "VMV3": "Ventromedial Visual 3 — peripheral visual field; medial ventral stream.",
    "4": "Primary motor cortex (M1) — execution of voluntary movement.",
    "3a": "Somatosensory area 3a — proprioceptive input from muscle spindles.",
    "3b": "Primary somatosensory cortex (S1, area 3b) — cutaneous touch.",
    "1": "Somatosensory area 1 — fine touch & texture.",
    "2": "Somatosensory area 2 — higher-order touch, shape from touch.",
    "5L": "Superior parietal area 5L — limb coordinates for reaching.",
    "5m": "Medial superior parietal area 5m — body schema.",
    "5mv": "Ventral 5m — leg/foot somatomotor association.",
    "6mp": "Posterior medial premotor — supplementary motor area (SMA proper).",
    "6ma": "Anterior medial premotor — pre-SMA; action selection and inhibition.",
    "6d": "Dorsal premotor — visually-guided reaching.",
    "6v": "Ventral premotor — orofacial action; grasp planning.",
    "6a": "Premotor area 6a — action planning.",
    "6r": "Premotor 6r — motor preparation.",
    "FEF": "Frontal Eye Field — voluntary saccades, spatial attention.",
    "PEF": "Premotor Eye Field — eye-movement planning adjacent to FEF.",
    "55b": "Area 55b — speech-motor / language production interface.",
    "SFL": "Superior Frontal Language area — speech motor and language production.",
    "A1": "Primary auditory cortex — frequency-tuned core auditory processing.",
    "MBelt": "Middle auditory belt — spectrotemporal sound features.",
    "LBelt": "Lateral auditory belt — complex sound and species-specific vocalizations.",
    "PBelt": "Parabelt auditory cortex — complex acoustic and speech sound integration.",
    "RI": "RI — auditory association near primary auditory cortex.",
    "A4": "Auditory area 4 — higher-order auditory association.",
    "A5": "Auditory area 5 — speech and complex sound perception.",
    "44": "Area 44 (Broca pars opercularis) — speech production, syntax (left); inhibitory control (right).",
    "45": "Area 45 (Broca pars triangularis) — semantic processing, lexical retrieval.",
    "47l": "Lateral 47 — semantic / conceptual control.",
    "47s": "Sulcal 47 — value-based decision making, semantic.",
    "47m": "Medial 47 — orbital reward valuation.",
    "p47r": "Posterior 47r — semantic control, language selection.",
    "a47r": "Anterior 47r — semantic / conceptual control.",
    "IFJa": "Inferior frontal junction anterior — cognitive control, task switching.",
    "IFJp": "Inferior frontal junction posterior — attention switching, response inhibition.",
    "IFSa": "Inferior frontal sulcus anterior — working memory, language selection.",
    "IFSp": "Inferior frontal sulcus posterior — visuospatial working memory.",
    "i6-8": "Intermediate 6/8 — frontal control, attention.",
    "s6-8": "Superior 6/8 — frontal control.",
    "8Av": "Ventral 8A — attentional control, working memory.",
    "8Ad": "Dorsal 8A — visuospatial working memory.",
    "8C": "Area 8C — cognitive control, abstract rules.",
    "8BL": "Lateral 8B — working memory and attention.",
    "8BM": "Medial 8B — error monitoring, self-evaluation.",
    "9-46d": "Dorsal 9/46 — DLPFC; manipulation in working memory.",
    "9-46v": "Ventral 9/46 — DLPFC; goal-directed cognitive control.",
    "46": "Area 46 — DLPFC core; working memory maintenance.",
    "p9-46v": "Posterior 9/46v — DLPFC; flexible rule encoding.",
    "9p": "Posterior 9 — top-down control, planning.",
    "9m": "Medial 9 — self-referential cognition, mentalizing.",
    "9a": "Area 9a — DLPFC; abstract reasoning.",
    "10d": "Dorsal frontopolar 10 — high-level planning, multitasking.",
    "10v": "Ventral frontopolar 10 — value-guided decisions.",
    "10pp": "Polar 10 — most-anterior frontopolar; introspection, future thinking.",
    "10r": "Rostral 10 — abstract relational reasoning.",
    "11l": "Lateral OFC — sensory value, reward.",
    "13l": "Medial OFC — outcome valuation.",
    "OFC": "Orbitofrontal cortex — value, hedonic experience, reward learning.",
    "pOFC": "Posterior OFC — primary reward and aversive value.",
    "AVI": "Anterior Ventral Insula — interoception, salience, emotional awareness.",
    "AAIC": "Anterior Agranular Insular Complex — emotional and visceral processing.",
    "MI": "Middle insula — interoception integration.",
    "PI": "Posterior insula — pain, body state.",
    "PoI1": "Posterior insula 1 — somatosensory/interoceptive.",
    "PoI2": "Posterior insula 2 — somatosensory/interoceptive.",
    "Ig": "Insular granular cortex — primary visceral/gustatory.",
    "FOP1": "Frontal opercular 1 — taste processing.",
    "FOP2": "Frontal opercular 2 — taste/somatosensory.",
    "FOP3": "Frontal opercular 3 — speech motor.",
    "FOP4": "Frontal opercular 4 — speech and language production.",
    "FOP5": "Frontal opercular 5 — speech motor / language.",
    "EC": "Entorhinal cortex — gateway to hippocampus; memory encoding/retrieval.",
    "PreS": "Presubiculum — spatial / head-direction signaling.",
    "H": "Hippocampus — declarative memory (episodic / spatial consolidation).",
    "ProS": "Prosubiculum — hippocampal output to cortex.",
    "PeEc": "Perirhinal cortex (PeEc) — object & item recognition memory.",
    "TGd": "Dorsal temporal pole — semantic and social knowledge.",
    "TGv": "Ventral temporal pole — semantic memory; person knowledge.",
    "TE1a": "Anterior superior temporal — semantic, language comprehension.",
    "TE1m": "Mid superior temporal — auditory/language convergence.",
    "TE1p": "Posterior superior temporal — speech sound and word recognition.",
    "TE2a": "Anterior inferior temporal — high-level visual category recognition.",
    "TE2p": "Posterior inferior temporal — object/category recognition.",
    "STSdp": "Posterior dorsal STS — biological motion, social perception.",
    "STSda": "Anterior dorsal STS — voice, social cognition.",
    "STSva": "Anterior ventral STS — speech and voice processing.",
    "STSvp": "Posterior ventral STS — speech and language comprehension.",
    "STGa": "Anterior superior temporal gyrus — auditory and semantic processing.",
    "STV": "Superior temporal visual area — TPJ-adjacent; biological motion.",
    "TPOJ1": "Temporo-parieto-occipital 1 — multimodal binding, language.",
    "TPOJ2": "Temporo-parieto-occipital 2 — theory of mind, social cognition.",
    "TPOJ3": "Temporo-parieto-occipital 3 — mentalizing, narrative comprehension.",
    "PSL": "Perisylvian Language area — language network hub.",
    "PFm": "PFm (inferior parietal lobule) — language and tool-use semantics.",
    "PFt": "PFt — tool use, action observation.",
    "PFcm": "PFcm — somatosensory association.",
    "PF": "PF (inferior parietal) — multimodal integration.",
    "PFop": "PF operculum — somatosensory/auditory integration.",
    "PGi": "PGi — angular gyrus; semantic processing, mentalizing, default mode.",
    "PGs": "PGs — angular gyrus; default-mode hub, autobiographical memory.",
    "PGp": "PGp — posterior angular gyrus; visuospatial / number cognition.",
    "IPS1": "Intraparietal sulcus 1 — visuospatial attention.",
    "IP0": "IP0 — intraparietal saliency map.",
    "IP1": "IP1 — number processing and quantity.",
    "IP2": "IP2 — anterior intraparietal; grasp planning.",
    "MIP": "Medial intraparietal — reach planning.",
    "LIPv": "Ventral lateral intraparietal — saccade planning.",
    "LIPd": "Dorsal lateral intraparietal — visual attention prioritization.",
    "VIP": "Ventral intraparietal — peripersonal space, multisensory.",
    "AIP": "Anterior intraparietal — grasp / hand-object interaction.",
    "7Pm": "Medial superior parietal — visuospatial transformation.",
    "7m": "7m — visuospatial / default-mode adjacent.",
    "7Am": "Medial 7A — reaching, body-in-space coordinates.",
    "7AL": "Lateral 7A — visuospatial.",
    "7PL": "Lateral 7P — visuospatial.",
    "POS1": "Parieto-occipital sulcus 1 — visuospatial / DMN-adjacent.",
    "POS2": "Parieto-occipital sulcus 2 — visuospatial / DMN-adjacent.",
    "PCV": "Precuneus visual — DMN, autobiographical memory, self.",
    "23c": "Mid-cingulate 23c — cognitive control, action monitoring.",
    "23d": "Mid-cingulate 23d — cognitive control.",
    "d23ab": "Dorsal posterior cingulate (23ab) — DMN, autobiographical memory.",
    "v23ab": "Ventral posterior cingulate (23ab) — DMN, value, self-relevance.",
    "31a": "Anterior 31 — DMN; integration of memory and self.",
    "31pd": "Dorsal posterior 31 — DMN.",
    "31pv": "Ventral posterior 31 — DMN.",
    "RSC": "Retrosplenial cortex — spatial memory, navigation, scene context.",
    "DVT": "Dorsal Visual Transitional — visual / DMN interface.",
    "ProM": "Promotor / cingulate — motor association.",
    "p32": "Posterior 32 — anterior DMN, value, self-relevance.",
    "a32pr": "Anterior 32pr — cognitive control, conflict monitoring.",
    "p32pr": "Posterior 32pr — cognitive control.",
    "d32": "Dorsal 32 — error / conflict monitoring.",
    "p24": "Posterior 24 — mid-cingulate; action-outcome monitoring.",
    "a24": "Anterior 24 — emotional regulation, salience.",
    "a24pr": "Anterior mid-cingulate (24pr) — cognitive control / salience.",
    "p24pr": "Posterior mid-cingulate (24pr) — action selection under conflict.",
    "24dd": "Cingulate motor area (dorsal) — voluntary action selection.",
    "24dv": "Cingulate motor area (ventral) — action selection.",
    "33pr": "Cingulate 33pr — cognitive / autonomic control.",
    "25": "Subgenual ACC (25) — emotion regulation, mood, autonomic.",
    "s32": "Subgenual 32 — affect; implicated in mood disorders.",
    "Pir": "Piriform cortex — primary olfactory processing.",
    "OP1": "Parietal operculum 1 (S2) — secondary somatosensory.",
    "OP2-3": "Parietal operculum 2/3 — vestibular, somatosensory.",
    "OP4": "Parietal operculum 4 — somatosensory/auditory.",
    "43": "Area 43 — orofacial sensorimotor; speech.",
    "52": "Area 52 — auditory parabelt and operculum.",
}


# ---------------------------------------------------------------------------
# Lifespan / model warmup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("loading TRIBE v2 model (this takes ~15s the first time)...")
    state["model"] = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=CACHE
    )
    state["pb"] = PlotBrainPyvista(mesh="fsaverage5", inflate="half", bg_map="sulcal")
    state["lock"] = asyncio.Lock()  # serialize predictions; one model instance

    log.info("loading HCP-MMP parcellation (first run downloads ~100 MB)...")
    state["fine_to_macro"] = await asyncio.to_thread(_build_fine_to_macro)
    state["fine_labels_lh"] = list(
        get_hcp_labels(mesh="fsaverage5", combine=False, hemi="left").keys()
    )
    state["fine_labels_rh"] = list(
        get_hcp_labels(mesh="fsaverage5", combine=False, hemi="right").keys()
    )
    log.info("HCP-MMP loaded: %d fine regions × 2 hemispheres", len(state["fine_labels_lh"]))
    macro_used = sorted(set(state["fine_to_macro"].values()))
    unmatched = [m for m in macro_used
                 if m not in MACRO_DESCRIPTIONS
                 and _norm_macro_key(m) not in _MACRO_DESCRIPTIONS_NORM]
    if unmatched:
        log.warning("macro keys with no description (will use generic fallback): %s", unmatched)
    else:
        log.info("all %d macro groups mapped to descriptions", len(macro_used))

    log.info("baking per-vertex ambient occlusion from mesh curvature...")
    state["ao_l"] = await asyncio.to_thread(
        _compute_ao, state["pb"]._mesh["left"]["coords"], state["pb"]._mesh["left"]["faces"]
    )
    state["ao_r"] = await asyncio.to_thread(
        _compute_ao, state["pb"]._mesh["right"]["coords"], state["pb"]._mesh["right"]["faces"]
    )
    log.info("AO baked (range L=%.2f..%.2f R=%.2f..%.2f)",
             state["ao_l"].min(), state["ao_l"].max(),
             state["ao_r"].min(), state["ao_r"].max())

    log.info("warming up encoders with a dummy phrase (one-time ~60s)...")
    try:
        await asyncio.to_thread(predict_signal, "the quick brown fox jumps over the lazy dog")
        log.info("warmup complete; encoders are resident in VRAM.")
    except Exception as e:
        log.warning("warmup failed (non-fatal): %s", e)

    log.info("ready on %s:%d", HOST, PORT)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HCP region helpers
# ---------------------------------------------------------------------------

def _compute_ao(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Bake per-vertex ambient occlusion from local mean curvature.

    Concave neighborhoods (sulci) get a low multiplier → darker; convex
    (gyri) stay bright. Computed once at startup and multiplied into the
    final per-vertex RGB at render time so cortical folds visibly *recede*
    even where the prediction signal is strong.
    """
    pv_faces = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces])
    mesh = pv.PolyData(verts, pv_faces)
    curv = np.asarray(mesh.curvature(curv_type="mean"), dtype=np.float32)

    # One pass of mean filtering over ring-1 neighbours: soften per-triangle
    # noise so AO reads as smooth shading rather than speckle.
    n = len(verts)
    sums = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    for u, v in ((a, b), (b, c), (c, a)):
        np.add.at(sums, u, curv[v])
        np.add.at(counts, u, 1)
        np.add.at(sums, v, curv[u])
        np.add.at(counts, v, 1)
    smoothed = sums / np.maximum(counts, 1)
    smoothed = 0.5 * curv + 0.5 * smoothed

    lo, hi = np.percentile(smoothed, 5), np.percentile(smoothed, 95)
    norm = np.clip((smoothed - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    # ao multiplier: concave (low curvature) → 0.55; convex → 1.0
    ao = 0.55 + norm * 0.45
    return ao.astype(np.float32)


def _build_fine_to_macro() -> dict[str, str]:
    """Map each HCP-MMP fine region (180 names) to its Glasser macro group."""
    fine = get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both")
    macro = get_hcp_labels(mesh="fsaverage5", combine=True, hemi="both")
    mapping: dict[str, str] = {}
    macro_sets = {m: set(v.tolist()) for m, v in macro.items()}
    for fname, fverts in fine.items():
        fset = set(fverts.tolist())
        best, best_overlap = None, 0
        for mname, mset in macro_sets.items():
            o = len(fset & mset)
            if o > best_overlap:
                best, best_overlap = mname, o
        mapping[fname] = best or "Lateral_Temporal_Cortex"
    return mapping


def _norm_macro_key(s: str) -> str:
    """Normalize macro-region key for lookup tolerance.

    mne returns slightly different separators across versions
    (spaces / underscores / "and" vs "&"). Strip everything to lowercase
    alphanumerics so we can match REGION_DESCRIPTIONS keys regardless.
    """
    return "".join(ch.lower() for ch in s if ch.isalnum())


_MACRO_DESCRIPTIONS_NORM: dict[str, str] = {
    _norm_macro_key(k): v for k, v in MACRO_DESCRIPTIONS.items()
}


def _describe_region(name: str) -> str:
    if name in REGION_DESCRIPTIONS:
        return REGION_DESCRIPTIONS[name]
    macro = state.get("fine_to_macro", {}).get(name)
    if macro:
        if macro in MACRO_DESCRIPTIONS:
            return MACRO_DESCRIPTIONS[macro]
        norm = _MACRO_DESCRIPTIONS_NORM.get(_norm_macro_key(macro))
        if norm:
            return norm
        # macro is known but unmapped — surface its name so it's at least
        # interpretable rather than a bare "—"
        return f"Cortical region in {macro.replace('_', ' ')}."
    return "Cortical region (HCP-MMP)."


def top_regions(signal: np.ndarray, k: int = 8) -> list[dict]:
    """Top-K HCP-MMP fine regions by |mean signal|, with hemisphere split."""
    left = summarize_by_roi(signal, hemi="left", mesh="fsaverage5")
    right = summarize_by_roi(signal, hemi="right", mesh="fsaverage5")
    labels_l = state["fine_labels_lh"]
    labels_r = state["fine_labels_rh"]

    rows: list[dict] = []
    for name, val in zip(labels_l, left):
        rows.append({"name": name, "hemi": "lh", "mean": float(val)})
    for name, val in zip(labels_r, right):
        rows.append({"name": name, "hemi": "rh", "mean": float(val)})
    rows.sort(key=lambda r: abs(r["mean"]), reverse=True)
    out = []
    for r in rows[:k]:
        r["description"] = _describe_region(r["name"])
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Prediction + GLB rendering
# ---------------------------------------------------------------------------

def predict_signal(phrase: str, reduction: str = "mean") -> tuple[np.ndarray, int]:
    model = state["model"]
    io_dir = WORKDIR / "io"
    io_dir.mkdir(parents=True, exist_ok=True)
    text_path = io_dir / "_ws_phrase.txt"
    text_path.write_text(phrase + "\n", encoding="utf-8")

    events = model.get_events_dataframe(text_path=str(text_path))
    preds, _ = model.predict(events=events, verbose=False)
    if preds.shape[0] == 0:
        raise ValueError("no segments kept; phrase is too short")

    reducer = {
        "mean": lambda a: a.mean(axis=0),
        "max": lambda a: a.max(axis=0),
        "first": lambda a: a[0],
        "last": lambda a: a[-1],
    }[reduction]
    return reducer(preds).astype(np.float32), int(preds.shape[0])


def signal_to_glb(signal: np.ndarray, cmap_name: str = "bwr") -> bytes:
    """Render the per-vertex signal onto fsaverage5 with sulcal-depth shading.

    Uses TRIBE's standard recipe (cortical_pv.py:122–128): a thresholded,
    transparency-aware colormap; alpha is then *baked* into the RGB by
    blending against a sulcal-depth-shaded gray background. The resulting
    mesh has visible gyrification everywhere, with the prediction signal
    only popping where it's strong.
    """
    pb: PlotBrainPyvista = state["pb"]

    # alpha_cmap=(threshold, scale): below threshold → transparent; ramp to
    # opaque between threshold and threshold+scale. For bwr/seismic this is
    # symmetric around the zero point.
    cmap = get_cmap(cmap_name, alpha_cmap=(0.5, 0.4))
    sm = get_scalar_mappable(signal, cmap, symmetric_cbar=True)

    verts_l = pb._mesh["left"]["coords"]
    verts_r = pb._mesh["right"]["coords"]
    faces_l = pb._mesh["left"]["faces"]
    faces_r = pb._mesh["right"]["faces"]
    bg_l = pb._mesh["left"]["bg_map"]
    bg_r = pb._mesh["right"]["bg_map"]
    n_l = len(verts_l)

    rgba = sm.to_rgba(signal)  # (N, 4)
    rgba_l, rgba_r = rgba[:n_l], rgba[n_l:]

    def _blend(rgba_h, bg_map_h, ao_h):
        bg_norm = (bg_map_h - bg_map_h.min()) / (bg_map_h.max() - bg_map_h.min() + 1e-8)
        # darker valleys (sulci), lighter ridges (gyri): 0.15..1.0 brightness
        bg_rgb = 1.0 - np.column_stack([0.15 + bg_norm * 0.85] * 3)
        a = rgba_h[:, 3:4]
        blended = a * rgba_h[:, :3] + (1.0 - a) * bg_rgb
        # multiply baked AO into both signal and background so deep sulci
        # darken even where the prediction is strong → visible cortical folds.
        return blended * ao_h[:, None]

    rgb_l = _blend(rgba_l, bg_l, state["ao_l"])
    rgb_r = _blend(rgba_r, bg_r, state["ao_r"])
    rgb = np.vstack([rgb_l, rgb_r])
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    verts = np.vstack([verts_l, verts_r]).astype(np.float32)
    faces = np.vstack([faces_l, faces_r + n_l])
    pv_faces = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces])

    mesh = pv.PolyData(verts, pv_faces)
    mesh.point_data["RGB"] = rgb_u8
    mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        inplace=True,
        consistent_normals=True,
    )

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, scalars="RGB", rgb=True, smooth_shading=True)
    pl.set_background("white")

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name
    pl.export_gltf(tmp_path)
    pl.close()
    glb = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return glb


# ---------------------------------------------------------------------------
# HTTP / WebSocket endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"ok": state.get("model") is not None}


@app.websocket("/ws")
async def ws(ws: WebSocket) -> None:
    await ws.accept()
    log.info("client connected: %s", ws.client)
    try:
        while True:
            raw = await ws.receive_text()
            phrase, reduction, cmap = raw, "mean", "bwr"
            stripped = raw.strip()
            if stripped.startswith("{"):
                try:
                    payload = json.loads(stripped)
                    phrase = payload["phrase"]
                    reduction = payload.get("reduction", "mean")
                    cmap = payload.get("cmap", "bwr")
                except Exception as e:
                    await ws.send_text(f"ERROR: bad JSON: {e}")
                    continue
            log.info("predict: %r (reduction=%s, cmap=%s)", phrase[:80], reduction, cmap)
            try:
                async with state["lock"]:
                    signal, n_seg = await asyncio.to_thread(predict_signal, phrase, reduction)
                    regions = await asyncio.to_thread(top_regions, signal, 8)
                    glb = await asyncio.to_thread(signal_to_glb, signal, cmap)
                await ws.send_text(json.dumps({
                    "phrase": phrase,
                    "n_segments": n_seg,
                    "signal_min": float(signal.min()),
                    "signal_max": float(signal.max()),
                    "glb_bytes": len(glb),
                    "top_regions": regions,
                }))
                await ws.send_bytes(glb)
                log.info("sent %d-byte glb (top regions: %s)",
                         len(glb), ", ".join(f"{r['name']}-{r['hemi']}" for r in regions[:3]))
            except Exception as e:
                log.exception("predict failed")
                await ws.send_text(f"ERROR: {e}")
    except WebSocketDisconnect:
        log.info("client disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
