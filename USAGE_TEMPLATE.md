# Model Usage System — Project Template

A small, portable blueprint for shipping a from-scratch ML model as an **installable
package with multiple usage surfaces**. Hand this file to any new model project (a
translator, a classifier, an OCR model, an ASR model) to keep a consistent shape:
**one inference core, reused by every surface.**

Derived from the Netra-NMT layout. It is task-agnostic — replace "translate" with
your model's verb (`transcribe`, `classify`, `generate`, `detect`) and the rest holds.

---

## 1. Principles

1. **One core, many surfaces.** A single high-level class (`NetraTranslator`) loads the
   model + tokenizer once and exposes *the* inference method. Every surface (API, CLI,
   web) is a thin adapter over that class. Never reimplement inference per surface.
2. **2–3 usage surfaces, web is optional.** Ship at minimum a **Python API** and a **CLI**.
   Add a **Web app + REST API** as an *optional install extra* — not a hard dependency.
3. **Weights live on the Hub; code lives in the wheel.** Heavy weights download + cache on
   first use (`~/.cache/huggingface`). Small, always-needed assets (tokenizer) ship inside
   the package so there's no network round-trip to start.
4. **Lazy, cached loading.** Construct the model once per process. The web app loads on
   startup (lifespan); the one-shot helper caches a module-level default; Streamlit uses
   `@st.cache_resource`.
5. **Config is data, not code.** A single `config.py` holds constants (repo id, filenames,
   task directions/labels) and a `@dataclass ModelConfig` that mirrors the exported
   `config.json`. Nothing hard-codes a hyperparameter outside it.
6. **Three ways to point at weights, one resolver.** A `resolve()` function returns
   `(weights, config, tokenizer)` paths from either (a) the Hub default, (b) an env-var
   override, or (c) a local export dir — so every surface gets the same behaviour for free.
7. **Decoding/inference options are first-class and shared.** Greedy / beam / sample (or
   your task's equivalents) live in one module and every surface exposes the same knobs.

---

## 2. Package layout

```
your_model/
├── __init__.py          # public exports: Model class, one-shot fn, config, __version__
├── config.py            # constants + ModelConfig dataclass (mirrors config.json)
├── model.py             # the nn.Module architecture
├── decoding.py          # inference strategies (greedy/beam/sample, or task equivalent)
├── weights.py           # resolve() → (weights, config, tokenizer) paths
├── translator.py        # ★ high-level core class + one-shot convenience fn
├── cli.py               #   Surface 2: argparse entry point
├── server.py            #   Surface 3 (optional): FastAPI app + uvicorn entry point
├── assets/
│   └── spm_32k.model    #   tokenizer — bundled in the wheel (force-include)
└── static/
    └── index.html       #   web UI served by the FastAPI app
pyproject.toml           # console scripts + optional [web]/[train] extras + force-include
streamlit_space/         # optional alt web deployment (HF Spaces) — reuses the package
scripts/                 # training / export / benchmarking (not shipped in the wheel)
```

---

## 3. The shared core (every surface depends on this)

### 3.1 `config.py` — constants + dataclass

```python
HF_REPO_ID = "Org/your-model"          # default Hub repo for released weights
WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME  = "config.json"
TOKENIZER_FILENAME = "spm_32k.model"

# Task vocabulary: directions for translation, labels for classification, etc.
DIRECTIONS = {"en2km": ("en", "km"), "km2en": ("km", "en")}
DEFAULT_DIRECTION = "en2km"

@dataclass
class ModelConfig:
    """Architecture + tokenizer config, mirrors the exported config.json."""
    vocab_size: int = 32000
    d_model: int = 512
    # ... special token ids must match the tokenizer ...

    @classmethod
    def from_json(cls, path): ...   # tolerant: ignores unknown keys
    def to_json(self, path): ...
```

### 3.2 `weights.py` — one resolver, three sources

```python
def resolve(repo_id=None, local_dir=None) -> tuple[Path, Path, Path]:
    # 1. local_dir given  → load model.safetensors + config.json from disk (no Hub)
    # 2. else             → hf_hub_download(repo_id or $YOUR_MODEL_REPO_ID or HF_REPO_ID)
    # 3. tokenizer        → prefer the copy bundled in the wheel; fall back to the Hub
    return weights_path, config_path, tokenizer_path
```

> The env-var override (`$YOUR_MODEL_REPO_ID`, `$..._LOCAL_DIR`, `$..._DEVICE`) is what lets
> the web app be configured at deploy time without changing any call site.

### 3.3 `translator.py` — ★ the core class

This is the single most important file. Everything else is an adapter.

```python
class NetraTranslator:
    def __init__(self, repo_id=None, local_dir=None, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights, config, tok = resolve(repo_id=repo_id, local_dir=local_dir)
        self.config = ModelConfig.from_json(config)
        self.sp = load_tokenizer(tok)
        self.model = build_model(self.config); load_weights(self.model, weights)
        self.model.to(device).eval()

    def translate(self, text, direction=DEFAULT_DIRECTION, mode="greedy",
                  beam_size=5, temperature=1.0, top_p=0.95, max_new_tokens=128) -> str:
        # validate args → preprocess → encode → decoding.<mode>() → postprocess
        ...

    def translate_batch(self, texts, **kw) -> list[str]:
        return [self.translate(t, **kw) for t in texts]

# module-level one-shot helper with a cached default instance
_DEFAULT = None
def translate(text, **kw):
    global _DEFAULT
    if _DEFAULT is None: _DEFAULT = NetraTranslator()
    return _DEFAULT.translate(text, **kw)
```

### 3.4 `__init__.py` — public surface

```python
from .config import DIRECTIONS, ModelConfig
from .model import NetraNMT
from .translator import NetraTranslator, translate
__version__ = "0.1.0"
__all__ = ["NetraTranslator", "translate", "NetraNMT", "ModelConfig", "DIRECTIONS", "__version__"]
```

---

## 4. Usage Surface 1 — Python API  *(required)*

Two entry points: a reusable class and a one-shot function.

```python
from your_model import NetraTranslator, translate

t = NetraTranslator()                                  # auto device; downloads weights once
t.translate("Hello, how are you?", direction="en2km")  # → "..."
t.translate_batch(["Good morning.", "See you."], direction="en2km")
t.translate("Hello", direction="en2km", mode="beam", beam_size=5)

translate("Hello", direction="en2km")                  # one-shot, caches a default
```

**Checklist**
- [ ] Class loads model+tokenizer once; `translate()` is pure given the instance.
- [ ] `device` auto-detects (`cuda`→`cpu`), overridable.
- [ ] A `*_batch` helper exists.
- [ ] A module-level one-shot fn caches a default instance.
- [ ] Bad `direction`/`mode` raise clear `ValueError`s listing valid choices.

---

## 5. Usage Surface 2 — CLI  *(required)*

A console script wired in `pyproject.toml`, built on `argparse`, with **three input modes**:
single string, file (line-per-line), and an interactive REPL fallback.

```bash
your-cli --text "Hello, how are you?"                          # single sentence
your-cli --text "..." --direction km2en --mode beam            # options
your-cli --file input.txt --output output.txt --direction en2km # batch a file
your-cli                                                        # interactive REPL
```

**Structure** (`cli.py`):
- `parse_args()` — mutually-exclusive `--text` / `--file`; shared `--direction`,
  `--mode`, `--beam-size`, `--temperature`, `--top-p`, `--max-new-tokens`; model-source
  flags `--repo-id`, `--local-dir`, `--device`.
- `main()` — build one `NetraTranslator`, then branch: `--text` → translate+print+timing;
  `--file` → `_translate_file()` with progress; else → `_interactive_repl()`.
- **REPL** — `:dir`, `:mode`, `:beam`, `:temp`, `:topp`, `:maxlen`, `:help`, `:quit`;
  catch per-line exceptions so one bad input doesn't kill the session; print ms latency.

```toml
[project.scripts]
your-cli = "your_model.cli:main"
```

**Checklist**
- [ ] One translator built once, reused for all lines.
- [ ] `--text` / `--file` mutually exclusive; no args → REPL.
- [ ] Same decoding knobs as the Python API (don't drift).
- [ ] Per-item errors caught and reported, not fatal.
- [ ] Prints latency — useful for quick benchmarking.

---

## 6. Usage Surface 3 — Web app + REST API  *(optional, install extra)*

FastAPI serving both a JSON API and a static two-pane UI, exposed as a `your-web` script.
Gated behind an optional dependency so the core install stays light.

```bash
pip install "your-model[web]"
your-web                       # http://127.0.0.1:8000  (UI + API)
your-web --port 8080 --device cpu
your-web --local-dir export    # load weights from a local export dir
```

```bash
curl -X POST http://127.0.0.1:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello", "direction": "en2km"}'
# {"translation": "...", "direction": "en2km"}
```

**Structure** (`server.py`):
- **Lifespan loader** — build the translator once on startup into module state; read
  `$..._REPO_ID / _LOCAL_DIR / _DEVICE` from the env (so `main()` just sets env from flags).
- **Pydantic models** — `TranslateRequest` validates `mode`/`beam_size`/`max_new_tokens`
  with `Field(pattern=..., ge=..., le=...)`; `TranslateResponse` is the typed output.
- **Routes** — `POST /api/translate`, `GET /api/health` (returns device + load status),
  `GET /` serves `static/index.html`, `app.mount("/static", ...)` for brand assets.
- **`main()`** — argparse → set env vars → `uvicorn.run("your_model.server:app", ...)`.

```toml
[project.optional-dependencies]
web = ["fastapi>=0.110", "uvicorn[standard]>=0.27"]

[project.scripts]
your-web = "your_model.server:main"
```

**Checklist**
- [ ] FastAPI + uvicorn are an *extra*, never a core dependency.
- [ ] Model loaded once via lifespan, not per request.
- [ ] Request schema validates ranges; empty input returns empty, not a 500.
- [ ] `/api/health` exposes device + readiness.
- [ ] Config comes from env vars so deploy time needs no code change.

### 6b. Alternative web deployment — Streamlit Space *(optional)*

For a zero-backend hosted demo (e.g. Hugging Face Spaces), a `streamlit_space/app.py` that
**imports the same `NetraTranslator`** and caches it with `@st.cache_resource`. It mirrors
the FastAPI UI (two panes, swap button) but is self-contained for the free CPU tier. Pick
*one* web surface per project — FastAPI for an API you control, Streamlit for a quick demo.

---

## 7. Packaging (`pyproject.toml`) essentials

```toml
[project]
dependencies = ["torch>=2.0", "<tokenizer>", "safetensors", "huggingface_hub", "numpy"]

[project.optional-dependencies]
web   = ["fastapi", "uvicorn[standard]"]          # Surface 3
train = ["datasets", "sacrebleu", "accelerate"]   # scripts/ only — never required to infer

[project.scripts]
your-cli = "your_model.cli:main"
your-web = "your_model.server:main"

# Ship non-Python assets inside the wheel (tokenizer, web UI, brand images):
[tool.hatch.build.targets.wheel.force-include]
"your_model/assets/spm_32k.model" = "your_model/assets/spm_32k.model"
"your_model/static/index.html"    = "your_model/static/index.html"
```

- **Core deps** = the minimum to run inference. Web and training are **extras**.
- **Bundle small assets**, download big weights. Tokenizer in the wheel; weights on the Hub.
- Build (`python -m build`) yields the `dist/*.whl` + `*.tar.gz` to publish to PyPI.

---

## 8. Quick checklist for a new model project

- [ ] `config.py`: repo id, filenames, task labels, `ModelConfig` dataclass.
- [ ] `weights.py`: one `resolve()` covering Hub default / env override / local dir.
- [ ] `translator.py`: core class (load-once) + one-shot cached helper.
- [ ] `__init__.py`: export the class, the helper, config, `__version__`.
- [ ] **Surface 1 — Python API** (required).
- [ ] **Surface 2 — CLI** (required): `--text` / `--file` / REPL, console script.
- [ ] **Surface 3 — Web + REST API** (optional): FastAPI under a `[web]` extra, `your-web`.
- [ ] Decoding/inference options identical across all surfaces.
- [ ] `pyproject.toml`: console scripts, optional extras, force-include assets.
- [ ] README "Usage" section with one block per surface (mirror §4–6 here).
```
