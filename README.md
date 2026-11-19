# paperless-vlm-ocr

A [Paperless-ngx](https://docs.paperless-ngx.com/) parser plugin that repairs the
text layer at ingestion time, before matching, workflows, full-text search and
the AI index see it. Paperless trusts a born-digital PDF's text layer and only
runs Tesseract on pages that have none, so split, letter-spaced or garbled
layers stay broken and their text is missing from search. The plugin keeps the
stock Tesseract parse, compares usable text layers with a fresh Tesseract read,
and transcribes the pages that do not pass with an OpenAI-compatible vision
model. The archive PDF is never modified; only `document.text` is replaced.

## How it works

```text
parse -> stock Tesseract parse (archive + fallback text, unchanged)
      -> per page: born-digital layer vs. Tesseract read (when the layer has text)
      -> gate: keep the layer | transcribe the page | keep the fallback
      -> rebuild document.text only if at least one page changed
```

Both texts are split into casefolded alphanumeric tokens. Runs of separately
spaced single characters are joined (`K a r l` becomes `karl`), so
letter-spaced layers still match, while wrongly split words (`Ko nto inha ber`)
do not. With `coverage` = shared tokens / reference tokens and `support` =
shared tokens / layer tokens, a page with at least `GATE_MIN_TOKENS` tokens on
both sides keeps its layer when

```
keep if coverage >= threshold
     or (coverage >= threshold - margin and support >= threshold)
```

`threshold` is `PAPERLESS_VLM_OCR_GATE_THRESHOLD`; `margin` and the tokenisation
rule are constants in `gate.py`. Otherwise:

| Page state | `shadow` (default) | `enforce` |
|---|---|---|
| layer passes the gate | `keep` | `keep` |
| layer fails the gate | `would_vlm`, layer kept | `vlm`, transcribed |
| fewer than `GATE_MIN_TOKENS` layer tokens (`empty_layer`) | `vlm`, transcribed | `vlm`, transcribed |
| reference read too short (`no_signal`) | `no_signal`, layer kept | `vlm`; the layer wins if it agrees with the model |

`shadow` is for testing. Pages with a usable layer are left alone, but every
page that would be sent to the model is logged.

Model text replaces the layer unless the page was `no_signal` and the two
agree. If a call fails or its output is rejected, the page falls back to the
Tesseract reference read (gated pages), the archive's text layer (pages that
never had one, or a Tesseract read when there is no archive) or the
born-digital layer (`no_signal`). A failed reference read keeps the layer. If
the plugin itself raises, it discards its work and keeps the parent parser's
result, so ingestion does not fail because of the plugin.

### Output checks

Each output is checked before it can replace a page:

- Loop: a unit of text that repeats many times in a row means the model is
  looping. Text before the line where the loop starts is kept as `loop_cut`
  when it is long enough compared with the expected page text (the ratio is a
  constant in `vlm.py`); otherwise the output is rejected as `repetition`.
- Truncated: `finish_reason=length` is rejected as `truncated`.
- Empty: output with no real text against a non-trivial fallback is `empty`.
- Runaway: output much longer than the fallback is `runaway`.

The loop and runaway limits are constants in `vlm.py`. `truncated`,
`repetition` and `empty` are retried, and `attempts` then covers both paid
calls. `runaway` and transport errors are not retried beyond the client's own
retries.

### Tall pages

A page above a fixed height-to-width ratio is rendered once and split into
tiles; the ratio and tile geometry are set in `pages.py`. Each tile's long side
is near `IMAGE_PX`. Seams fall on blank row bands near the nominal boundary, or
on the least-inked row. Adjacent tiles share one text line, which is removed
when the tiles are joined. A failed or rejected tile rejects the whole page.

### Time budget

Each parse has a time budget counted from the start of `parse()`, which
includes OCRmyPDF and the reference reads. `DOC_BUDGET_S` sets it, capped at the
Celery task limit minus a safety margin (constant in `vlm.py`). If the budget
is used up before the reference reads, they are skipped and pages with a layer
are kept with reason `budget`. Before every model call, including each tile and
the retry, the remaining time is checked. Per-call timeouts and transport
retries are capped so they cannot exceed it, and the retry is skipped when the
remaining time cannot cover a full timeout.

## Installation

The release is a data-only OCI image (built `FROM scratch`) containing exactly
`paperless_vlm_ocr/` and `paperless_vlm_ocr-<version>.dist-info/`. It is never
run. Mount it read-only on a directory listed in `PYTHONPATH`, and Paperless
finds the parser entry point through `importlib.metadata`. The Kubernetes
image-volume variant needs image-volume support (app-template 5.3 requires
Kubernetes 1.33+ for `persistence.type: image`). The Paperless image must ship
`openai`, Pillow, poppler (`pdfinfo`, `pdftotext`, `pdftoppm`) and `tesseract`;
the official image does.

### Kubernetes (app-template)

```yaml
controllers:
  main:
    containers:
      main:
        env:
          PYTHONPATH: /usr/src/paperless/vlm-ocr
          PAPERLESS_VLM_OCR_ENABLED: "true"
          PAPERLESS_VLM_OCR_GATE_MODE: shadow
          PAPERLESS_VLM_OCR_ENDPOINT:
            valueFrom:
              secretKeyRef: {name: paperless-secrets, key: llm-endpoint}
          PAPERLESS_VLM_OCR_API_KEY:
            valueFrom:
              secretKeyRef: {name: paperless-secrets, key: llm-api-key}
          PAPERLESS_VLM_OCR_MODEL:
            valueFrom:
              secretKeyRef: {name: paperless-secrets, key: llm-model}

persistence:
  vlmocr:
    type: image
    image: ghcr.io/anokfireball/paperless-vlm-ocr:0.3.0
    globalMounts:
      - path: /usr/src/paperless/vlm-ocr
        readOnly: true
```

### Plain Docker

For plain Docker, bind-mount the `site/` tree:

```sh
docker run --rm \
  -v "$PWD/site:/usr/src/paperless/vlm-ocr:ro" \
  -e PYTHONPATH=/usr/src/paperless/vlm-ocr \
  -e PAPERLESS_VLM_OCR_ENABLED=true \
  -e PAPERLESS_VLM_OCR_ENDPOINT=https://llm.example/v1 \
  -e PAPERLESS_VLM_OCR_API_KEY=... \
  -e PAPERLESS_VLM_OCR_MODEL=... \
  ghcr.io/paperless-ngx/paperless-ngx:3.5.1
```

Check that Paperless sees the plugin:

```sh
docker run --rm --read-only --tmpfs /tmp \
  -v "$PWD/site:/usr/src/paperless/vlm-ocr:ro" \
  -e PYTHONPATH=/usr/src/paperless/vlm-ocr \
  --entrypoint python3 ghcr.io/paperless-ngx/paperless-ngx:3.5.1 \
  -c "from importlib.metadata import entry_points, version; print([e.value for e in entry_points(group='paperless_ngx.parsers')], version('paperless-vlm-ocr'))"
```

## Configuration

All settings are environment variables with the `PAPERLESS_VLM_OCR_` prefix.
They are read when the parser is selected and again when it is constructed, so
restart the workers after a change. The parser is only chosen when `ENABLED` is
true, `ENDPOINT`, `API_KEY` and `MODEL` are all set, the Paperless minor is
supported, and the file type is one the Tesseract parser handles.

Defaults come from `site/paperless_vlm_ocr/config.py`, in its dataclass and its
`from_env` method. Numeric rows show `from config.py` instead of a value, so a
calibration change cannot make this table stale.

| Variable | Default | Meaning |
|---|---|---|
| `PAPERLESS_VLM_OCR_ENABLED` | `false` | `true` or `1` enables the plugin |
| `PAPERLESS_VLM_OCR_ENDPOINT` | *(unset)* | OpenAI-compatible base URL, usually ending in `/v1` |
| `PAPERLESS_VLM_OCR_API_KEY` | *(unset)* | API key for that endpoint |
| `PAPERLESS_VLM_OCR_MODEL` | *(unset)* | vision model name |
| `PAPERLESS_VLM_OCR_GATE_MODE` | `shadow` | `shadow` logs candidates; `enforce` replaces them (any other value means `shadow`) |
| `PAPERLESS_VLM_OCR_GATE_THRESHOLD` | from `config.py` | agreement threshold `t` in the gate rule |
| `PAPERLESS_VLM_OCR_GATE_MIN_TOKENS` | from `config.py` | minimum tokens for a layer or reference read to be judged |
| `PAPERLESS_VLM_OCR_TIMEOUT_S` | from `config.py` | per-request timeout in seconds |
| `PAPERLESS_VLM_OCR_DOC_BUDGET_S` | from `config.py` | time budget per document, from the start of `parse()` |
| `PAPERLESS_VLM_OCR_CONCURRENCY` | from `config.py` | parallel model requests per document |
| `PAPERLESS_VLM_OCR_MAX_PAGES` | from `config.py` | documents with more pages are skipped untouched (warning) |
| `PAPERLESS_VLM_OCR_IMAGE_PX` | from `config.py` | long side of each page/tile image sent to the model |
| `PAPERLESS_VLM_OCR_DECISION_LOG` | `<DATA_DIR>/vlm-ocr/decisions.jsonl` | JSONL decision log path |

`TIMEOUT_S` and `DOC_BUDGET_S` accept fractional seconds.

## Observability

### Logs

All plugin messages use the `paperless.parsing.vlm_ocr` logger and are written
to `paperless.log` and the worker's stdout. The parent parser keeps its own
`paperless.parsing.tesseract` logger, so stock messages such as
`pdftotext exited 0` are unchanged. `sha` is the first 12 hex digits of the file
checksum. `file` appears only on start, summary and skip lines.

Angle brackets in these examples mark values that depend on the document or the
configuration:

```
# one start line per document
vlm_ocr start file=statement.pdf sha=a1b2c3d4e5f6 mime=application/pdf pages=5 archive=yes gate_mode=shadow threshold=<t> model=vision-model

# one line per page, after all model calls
vlm_ocr page 2/5 sha=a1b2c3d4e5f6 kind=gated decision=keep coverage=<c> support=<s> threshold=<t> layer_tokens=<n> ref_tokens=<n> source=layer reason=- tiles=<n> loop_at=-
vlm_ocr page 1/5 sha=a1b2c3d4e5f6 kind=empty_layer decision=vlm coverage=- support=- threshold=<t> layer_tokens=<n> ref_tokens=- source=vlm reason=- vlm_vs_fallback_coverage=<c> vlm_vs_fallback_support=<s> tiles=<n> loop_at=-
vlm_ocr page 4/5 sha=a1b2c3d4e5f6 kind=gated decision=vlm coverage=<c> support=<s> threshold=<t> layer_tokens=<n> ref_tokens=<n> source=tesseract reason=APIConnectionError tiles=<n> loop_at=-

# one line per model call: accepted (INFO), rejected or failed (WARNING)
vlm_ocr llm page=1 sha=a1b2c3d4e5f6 model=vision-model host=api.example.test attempts=<n> finish_reason=stop tokens_in=<in> tokens_out=<out> seconds=12.3 image=<w>x<h> bytes=<bytes> chars=47 response_id=chatcmpl-abc request_id=req-123 tiles=<n> reason=- loop_at=-
vlm_ocr llm page=2 sha=a1b2c3d4e5f6 model=vision-model host=api.example.test attempts=<n> finish_reason=stop tokens_in=<in> tokens_out=<out> seconds=12.3 image=<w>x<h> bytes=<bytes> chars=167 response_id=chatcmpl-abc request_id=req-123 rejected reason=repetition fallback_chars=<n> tiles=<n> loop_at=-
vlm_ocr llm page=3 sha=a1b2c3d4e5f6 model=vision-model host=api.example.test failed error=APIStatusError status=429 request_id=req-999 seconds=3.5 tiles=<n>

# one summary per document
vlm_ocr summary file=statement.pdf sha=a1b2c3d4e5f6 pages=5 keep=2 vlm=1 fallback=2 would_vlm=0 no_signal=0 gate_mode=shadow threshold=<t> coverage_min=<c> coverage_mean=<c> support_min=<s> support_mean=<s> sources=LVLTT content=assembled tokens_in=<in> tokens_out=<out> seconds=18.0 budget_exhausted=no

# a document skipped before any work
vlm_ocr skipped file=archive.pdf sha=a1b2c3d4e5f6 pages=<n> max_pages=<limit>
```

- `decision` is the gate verdict, `source` is where the page's text came from
  (`layer`, `vlm`, `tesseract`), and `reason` is the final outcome: `loop_cut`,
  `budget`, a rejection, a fallback failure or the exception class of a failed
  call.
- `coverage`/`support` are `-` without a reference read. `vlm_vs_fallback_*`
  appear only after a model call and compare the output with the text it
  replaced. `tiles` counts the page's tiles, and `loop_at` is the loop offset in
  that call's text.
- In the summary, `sources` is one letter per page (`L` layer, `V` model, `T`
  Tesseract). `content` says whether `document.text` was rewritten
  (`assembled`) or left as the parent produced it (`parent`).
  `budget_exhausted=yes` means at least one page stopped on the budget.

Warnings also cover a plugin disabled by configuration or version (once per
process), a failed reference read, a failed fallback read and an unwritable
decision log.

### Decision log

One JSON object per page is appended to the decision log, whether or not
anything changed. A failed write is a warning, never a parse failure. Use
`sha256` to map a record back with `Document.objects.filter(checksum=...)`.

| Group | Fields |
|---|---|
| Document | `ts`, `sha256`, `mime`, `file`, `page`, `pages`, `plugin_version`, `gate_mode` |
| Gate | `kind`, `decision`, `coverage`, `support`, `threshold`, `layer_tokens`, `ref_tokens`, `source`, `reason`, `fallback_reason` |
| Model call | `model`, `attempts`, `finish_reason`, `vlm_seconds`, `input_tokens`, `output_tokens`, `payload_bytes`, `image_px`, `response_id`, `request_id`, `error_status` |
| Text | `vlm_chars`, `fallback_chars`, `layer_chars`, `vlm_vs_fallback_coverage`, `vlm_vs_fallback_support`, `tiles`, `loop_at` |

`fallback_reason` is a rejection reason, `budget`, `ref_error` or the exception
class of the failed call (with its HTTP status in `error_status`). `reason` is
the final outcome. `attempts` counts the first call plus the retry and
transport retries, and the ids identify the last call upstream.

The API key, the endpoint path/query/userinfo (only the host is logged),
request bodies and transcriptions never appear in the logs or the decision
log. Failures carry only the exception class, HTTP status and request id.

## Offline gate scan

`scan.py` runs the gate on PDFs without Django, a database or a vision model,
always in enforce mode, so you can preview which pages would be routed:

```sh
PYTHONPATH=site python3 -m paperless_vlm_ocr.scan \
  --lang deu+eng --jobs 8 --out results.jsonl *.pdf
```

It writes one JSONL record per page (`file`, `page`, `layer_tokens`,
`ref_tokens`, `coverage`, `support`, `decision`), where `decision` is `keep`,
`vlm`, `no_signal` or `empty_layer`. Unreadable files or pages produce
`decision: "error"` with an `error` message. `--threshold` and `--min-tokens`
have their own defaults; set them to the values your deployment uses. `--lang`
defaults to `deu+eng`, `--jobs` to a single worker and `--out` to
`results.jsonl`. Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`) and Tesseract must
be on `PATH`.

## Compatibility & upgrades

`SUPPORTED_PAPERLESS` in `site/paperless_vlm_ocr/__init__.py` lists the accepted
`(major, minor)` versions. `VERIFIED_PAPERLESS` above it names the Paperless
image CI verifies against. On any other minor `score()` returns `None`, warns
once and disables the plugin, so a Paperless upgrade falls back to stock
Tesseract. The plugin's score is higher than the built-in Tesseract parser's,
so it wins whenever it is eligible.

Upgrade order: merge and deploy the plugin release that lists the new minor
before upgrading Paperless. Listing a new minor keeps the old ones, so that
release still runs on the version being replaced and there is no gap without
the plugin.

## Development & releasing

Unit tests need Python 3.12, `pytest` and `openai`; the smoke test and CI need
Docker. Run the tests with the plugin tree on `PYTHONPATH`:

```sh
PYTHONPATH=site pytest -q tests
```

CI (`.github/workflows/ci.yaml`) runs three jobs:

1. test: no tab characters under `site/`, the three version places agree, unit
   tests.
2. paperless: against `VERIFIED_PAPERLESS`, checks the entry point,
   distribution version and supported version; runs unit tests inside the image
   when `pytest` is available; and does a smoke parse whose unreachable
   endpoint proves the Tesseract fallback, the archive and the decision log.
3. image: on `main` only, and only when `v<version>` is not tagged yet. Builds
   and pushes `ghcr.io/anokfireball/paperless-vlm-ocr:<version>` for
   `linux/amd64` and `linux/arm64`, then creates the tag and a GitHub release
   with generated notes. Re-merging a released version publishes nothing.

The version lives in three places that must move together: `__version__` in
`site/paperless_vlm_ocr/__init__.py`, the
`site/paperless_vlm_ocr-<version>.dist-info/` directory name, and the `Version:`
line in its `METADATA`.

Renovate tracks `VERIFIED_PAPERLESS` through the `# renovate:` regex-manager
annotation above it. For a patch of an already supported minor the
post-upgrade script changes nothing, and the in-image CI proves the new image.
For a new minor it adds the minor to `SUPPORTED_PAPERLESS` and bumps the
plugin's patch version in all three places, so merging that pull request
publishes a new image.
