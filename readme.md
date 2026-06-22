# Khmer ID Card Parser

A lightweight named-entity recognition model that extracts structured fields from raw OCR text of Cambodian national ID cards.

```json
{
  "id_number": "443322563",
  "name_khmer": "រ៉ា ចរិយា",
  "name_latin": "RA CHARIYA",
  "date_of_birth": "03.12.1962",
  "gender": "ស្រី",
  "height_cm": 172,
  "place_of_birth": { "commune": "សង្កាត់ផ្សារថ្មីទី ២", "district": "ខណ្ឌដូនពេញ", "province": "រាជធានីភ្នំពេញ" },
  "address":        { "village": "ត្រោក", "commune": "ឃុំសំបួរ", "district": "ស្រុករមាសហែក", "province": "ខេត្តស្វាយរៀង" },
  "validity":       { "issue_date": "៣១.០៨.២០១៧", "expiry_date": "៣១.០៨.២០២៧" },
  "distinguishing_marks": "ស្នាមសាក់រូបទេពតា",
  "mrz": ["IDKHM4433225638<<<<<<<<<<<<<<<", "6212036F2708311KHM<<<<<<<<<<<4", "RA<<CHARIYA<<<<<<<<<<<<<<<<<<<"]
}
```

---

## Project layout

```
Khmer-ID-Parser/
├── khmer_id_parser/          Installable package (one inference core, reused by every surface)
│   ├── __init__.py           Public exports: KhmerIDParser, parse, ModelConfig, __version__
│   ├── config.py             Constants + ModelConfig dataclass (mirrors checkpoint config)
│   ├── model.py              CharCNN + BiLSTM + CRF architecture
│   ├── postprocess.py        BIO spans → structured output schema
│   ├── weights.py            resolve() + load_checkpoint()
│   ├── parser.py             ★ KhmerIDParser core class + one-shot parse() helper
│   ├── cli.py                CLI (the `khmer-id-parse` console script)
│   └── assets/               Bundled weights + vocabularies (shipped in the wheel)
├── scripts/                  Training & data tooling (not shipped in the wheel)
│   ├── dataset.py            On-the-fly augmentation, vocab building, DataLoader
│   ├── train.py              Training loop with step/epoch metrics logging
│   ├── build_dataset.py      Build train/val JSONL from structured records
│   ├── generate_data.py      Gemini-based synthetic name/mark generation
│   └── plot_metrics.py       Generate training graphs from the metrics JSON
├── checkpoints/              Training outputs (weights, metrics, plots)
├── pyproject.toml
└── requirements.txt
```

---

## How it works

### 1 — Data format

Each line in your `.jsonl` training file is one ID card record:

```jsonl
{"nid": "443322563", "name_kh": "រ៉ា ចរិយា", "name_en": "RA CHARIYA", "dob": "03.12.1962", "gender": "ស្រី", "height": "172", "pob": ["ភូមិ២", "សង្កាត់ផ្សារថ្មីទី ២", "ខណ្ឌដូនពេញ", "រាជធានីភ្នំពេញ"], "addr": ["ត្រោក", "ឃុំសំបួរ", "ស្រុករមាសហែក", "ខេត្តស្វាយរៀង"], "issue_date": "31.08.2017", "exp_date": "31.08.2027", "mark": "ស្នាមសាក់រូបទេពតា", "mrz1": "IDKHM4433225638<<<<<<<<<<<<<<<", "mrz2": "6212036F2708311KHM<<<<<<<<<<<4", "mrz3": "RA<<CHARIYA<<<<<<<<<<<<<<<<<<<"}
```

Field reference:

| Key | Description |
|-----|-------------|
| `nid` | 9-digit national ID number |
| `name_kh` | Full name in Khmer script |
| `name_en` | Full name in Latin (as printed on card) |
| `dob` | Date of birth `DD.MM.YYYY` |
| `gender` | `ប្រុស` (male) or `ស្រី` (female) |
| `height` | Height in cm (numeric string) |
| `pob` | `[village, commune, district, province]` — place of birth |
| `addr` | `[village, commune, district, province]` — current address |
| `issue_date` | Card issue date `DD.MM.YYYY` |
| `exp_date` | Card expiry date `DD.MM.YYYY` |
| `mark` | Distinguishing marks (free text) |
| `mrz1/2/3` | Three MRZ lines |

### 2 — On-the-fly augmentation (`dataset.py`)

Each record is rendered into a card text string with BIO tags at `__getitem__` time, so every training epoch sees different augmented variants:

- **Whole-field drop** (2%) — randomly omits a field entirely
- **Character drop** (1%) — drops individual characters to simulate OCR dropout
- **Space removal** (10%) — removes spaces, common in Khmer OCR output
- **Noise word injection** (5%) — inserts random ASCII/Khmer tokens between fields
- **Date separator variation** — `.` or `:` randomly
- **Prefix stripping** (30%) — removes administrative prefixes (`ឃុំ`, `ស្រុក`, `ខេត្ត`, `សង្កាត់`, `ខណ្ឌ`, `រាជធានី`) to teach the model to recognise values with or without them
- **Unit suffix variation** — height unit randomly varies between `ស.ប`, `ស.ម`, `ផ.ម`

### 3 — Model architecture (`model.py`)

```
Input characters
       │
  CharEmbedding  (vocab_size × 64)
       │
  CharCNN  ── 3 parallel Conv1d, kernels [3, 5, 7], 64 filters each
              concat → 192-d per character
       │
  BiLSTM  ── 2 layers, 256 hidden per direction → 512-d
       │
  Linear  ── 512 → (num_tags − 1) emission scores
       │
  CRF  ── Viterbi decode, enforces valid BIO transitions
```

**Why `num_tags − 1` for the CRF?**
`TagVocab` reserves index 0 for `<PAD>`. The CRF has no concept of padding — it would freely predict tag-0 at real positions (~74% of the time at random initialisation). Excluding `<PAD>` from the CRF's tag space and using a ±1 offset on inputs/outputs fixes this.

### 4 — Training (`train.py`)

| Setting | Value |
|---------|-------|
| Optimiser | AdamW |
| Learning rate | 0.001 |
| LR schedule | Cosine annealing |
| Batch size | 64 |
| Epochs | 5 |
| Grad clip | 5.0 |
| Weight decay | 0.01 |
| CRF loss | `token_mean` (stable across sequence lengths) |

Training logs two levels of metrics to `checkpoints/training_metrics.json`:
- **Step level** — average loss every 50 steps, with global step, LR, and wall time
- **Epoch level** — train loss, val loss, val span-level F1, precision, recall

Best checkpoint is saved by **val F1** (not val loss).

---

## Setup

```bash
git clone https://github.com/your-username/khmer-id-parser
cd khmer-id-parser

python -m venv myenv
# Windows:
myenv\Scripts\activate
# macOS / Linux:
source myenv/bin/activate

# Inference only (installs the package + console script):
pip install -e .

# With training / data-generation tooling (scripts/):
pip install -e ".[train]"
```

The package bundles the trained weights and vocabularies, so inference works
immediately after install — no separate download step.

---

## Training

```bash
# Place your data files in the project root:
#   train_clean.jsonl  — training records
#   val_clean.jsonl    — validation records

pip install -e ".[train]"
python scripts/train.py
```

Checkpoints and metrics are saved to `checkpoints/`. To make a freshly trained
model the package default, copy it into the bundled assets:

```bash
cp checkpoints/khmer_id_parser_v2.pth khmer_id_parser/assets/
```

### Plot training curves

```bash
python scripts/plot_metrics.py
# → checkpoints/step_loss.png
# → checkpoints/epoch_losses.png
# → checkpoints/val_f1.png
# → checkpoints/lr_schedule.png
# → checkpoints/training_summary.png
```

![Training summary](checkpoints/training_summary.png)

---

## Inference

### Command-line  (`khmer-id-parse`)

```bash
# From a text file — pretty JSON (default)
khmer-id-parse --file card.txt

# From a text file — save output to a file
khmer-id-parse --file card.txt --output results/card.json

# Inline text (use $'...' in bash for literal newlines)
khmer-id-parse --text $'443322563\nគោន្តនាម...'

# Pipe from an OCR tool
tesseract card.png stdout | khmer-id-parse --stdin

# Output formats
khmer-id-parse --file card.txt --format json      # pretty JSON (default)
khmer-id-parse --file card.txt --format compact   # single-line JSON
khmer-id-parse --file card.txt --format tags      # character-level BIO alignment

# Output file extension auto-sets format
khmer-id-parse --file card.txt --output out.json    # pretty JSON
khmer-id-parse --file card.txt --output out.jsonl   # compact JSON
khmer-id-parse --file card.txt --output out.txt     # BIO tags

# Use a specific checkpoint (otherwise: $KHMER_ID_PARSER_CHECKPOINT, then bundled weights)
khmer-id-parse --file card.txt --checkpoint checkpoints/khmer_id_parser_v1.pth
```

### Python API

```python
from khmer_id_parser import KhmerIDParser, parse

parser = KhmerIDParser()               # bundled weights, auto device (cuda→cpu)

# Full structured dict
result = parser.parse(ocr_text)
print(result["name_latin"])            # "RA CHARIYA"
print(result["address"]["province"])   # "ខេត្តស្វាយរៀង"

# Raw entity spans (before post-processing)
spans = parser.extract_spans(ocr_text)
# {"ID_NUM": ["443322563"], "NAME_KH": ["រ៉ា ចរិយា"], "MRZ": [...], ...}

# Per-character BIO tags
tags = parser.tag(ocr_text)
# ["B-ID_NUM", "I-ID_NUM", ..., "B-NAME_KH", ...]

# Batch of cards
results = parser.parse_batch([text1, text2, text3])

# One-shot helper — caches a default parser, no explicit construction
parse(ocr_text)                        # → structured dict
```

Point at a different checkpoint either per-instance (`KhmerIDParser("other.pth")`)
or globally via the `KHMER_ID_PARSER_CHECKPOINT` / `KHMER_ID_PARSER_DEVICE`
environment variables.

### Extracted fields

| Output key | Entity tag | Notes |
|------------|------------|-------|
| `id_number` | `ID_NUM` | 9-digit string |
| `name_khmer` | `NAME_KH` | Khmer script |
| `name_latin` | `NAME_EN` | Latin transliteration |
| `date_of_birth` | `DOB` | Normalised to ASCII digits |
| `gender` | `GENDER` | Khmer script |
| `height_cm` | `HEIGHT` | Parsed to `int` |
| `place_of_birth.commune` | `POB_COMM` | |
| `place_of_birth.district` | `POB_DIST` | |
| `place_of_birth.province` | `POB_PROV` | |
| `address.village` | `ADDR_VILL` | |
| `address.commune` | `ADDR_COMM` | |
| `address.district` | `ADDR_DIST` | |
| `address.province` | `ADDR_PROV` | |
| `validity.issue_date` | `ISSUE_DATE` | Khmer numerals preserved |
| `validity.expiry_date` | `EXP_DATE` | Khmer numerals preserved |
| `distinguishing_marks` | `MARKS` | Free text |
| `mrz` | `MRZ` | List of 3 strings |

---

## Extending the model

**More training data** is the highest-leverage improvement. The model is trained entirely on synthetically rendered records — real annotated scans will close the gap between clean synthetic text and noisy real-world OCR output.

**Expanding place names** — `dataset.py` uses the place names directly from your `.jsonl` records. The wider the variety of provinces, districts, and communes in your data, the more robust the address extraction will be.

**Changing kernel sizes or hidden dimensions** — edit the constants at the top of `scripts/train.py`. The config is saved into the checkpoint, so `khmer_id_parser` reconstructs the correct architecture automatically (via `ModelConfig.from_dict`).

**Resuming training** — `training_metrics.json` is append-safe. If you stop mid-run and restart with the same `checkpoints/` directory, metrics accumulate rather than being overwritten.

---

## License

MIT
