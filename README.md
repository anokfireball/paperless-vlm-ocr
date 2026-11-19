# paperless-vlm-ocr

A [Paperless-ngx](https://docs.paperless-ngx.com/) parser plugin that repairs the
text layer at ingestion time, before matching, workflows and indexing.

Paperless normally trusts a born-digital PDF's text layer and only runs Tesseract
on pages without one. Broken layers (split or letter-spaced bank statements,
garbled archive conversions) therefore stay broken, while scans, receipts and
handwriting get plain Tesseract output.

This plugin keeps the stock Tesseract flow, then per page:

1. Reads the born-digital layer (if any) and renders the original page.
2. Runs Tesseract on the rendered page as a reference read.
3. Keeps the layer only when it agrees with that reference; otherwise the page is
   transcribed by an OpenAI-compatible vision model.
4. Falls back to Tesseract (or the archive's layer) whenever the model output is
   unusable, on API errors, or in `shadow` mode.

The archive PDF is never changed: it keeps OCRmyPDF's Tesseract text layer and
stays the fallback for search. Only `document.text` is replaced.

## Gate rule

Tokens are NFKC-casefolded alphanumeric runs; runs of three or more
single-character tokens are joined (`K a r l` → `karl`), so letter-spaced layers
still match, while wrongly split words (`Ko nto inha ber`) stay split and fail.

For a page with at least `GATE_MIN_TOKENS` layer tokens and a reference read with
at least `GATE_MIN_TOKENS` tokens, with

- `coverage` = shared tokens / reference tokens,
- `support` = shared tokens / layer tokens,

the layer is kept iff

```
coverage >= threshold or (coverage >= threshold - 0.1 and support >= threshold)
```

Otherwise:

| `GATE_MODE` | action |
|---|---|
| `shadow` | keep the stored layer, log `would_vlm` |
| `enforce` | transcribe the page with the vision model |

Pages with too few layer tokens are `empty_layer` and always go to the vision
model (in both modes). Pages whose reference read is too short are `no_signal`;
in `enforce` they are transcribed, and the model text replaces the layer unless
the layer agrees with it.

## Output checks

Model output is checked before it replaces the layer:

- A **periodic run** is a unit of 1-120 characters repeated at least 8 times
  and at least 200 characters long; it is degenerate generation, not content.
  The detector works inside one long line and across lines. The text before the
  line containing the loop start is kept as `loop_cut` when the fallback is at
  least 200 characters and that prefix covers at least 60% of it (for one tile,
  60% of that tile's 1/N share of the page fallback); otherwise the call is
  rejected as `repetition`.
- `finish_reason=length` is `truncated`.
- Fewer than 20 characters against a fallback of at least 50 is `empty`.
- More than `2.8 * fallback + 300` characters (fallback at least 200) is a
  `runaway` expansion.

A call rejected as `truncated`, `repetition` or `empty` is retried once for that
page only; `attempts` and the token counts then cover both calls. `runaway`
rejections and exceptions are not retried (transport retries stay in the
client).

Each parse has a wall-clock budget that counts from the start of `parse()`, so
OCRmyPDF and the reference reads are included, and that is capped by the Celery
task limit minus 120 s for post-processing:
`deadline = start + min(DOC_BUDGET_S, CELERY_TASK_TIME_LIMIT - 120)`, with
`DOC_BUDGET_S` defaulting to 900 s and the task limit to 1800 s. When the budget
is already exhausted, Tesseract reference reads are skipped and pages with a
layer are kept (`decision=keep`, reason `budget`). Before every model call -
first call, retry and each tile - the remaining budget is checked; when it is
exhausted the page falls back with reason `budget` instead of calling. Per-call
timeouts and transport retries are capped so they cannot overrun the budget,
and the extra retry is skipped when less than one timeout is left.

### Tall pages

A rendered page with height/width > 2.5 is cut into
`ceil((height / width) / 2)` tiles. The page is rasterised once so that each
tile's long side is about `IMAGE_PX` (full page height ~= tiles * `IMAGE_PX`,
capped at 10000 px). Each seam goes inside whitespace: within 6% of the tile
height of the nominal boundary, a blank row run of at least 8 px (row ink <= 12
of 255) is used, and the cut is the blank row nearest the boundary. If the band
has no blank run, the least-inked row is used. The next tile starts on the cut
row and the previous tile keeps the first complete text line below it, so
exactly one whole line is shared across the seam; tiles are transcribed
sequentially (pages stay parallel). `join_tiles` removes that line: up to two
leading lines of the next tile are dropped when they equal (after whitespace
normalisation), are fuzzy-equal (difflib ratio >= 0.85), or one is a prefix of
the other; the previous tile's version wins unless the next tile's line is
strictly longer. A failed or rejected tile rejects the whole page with that
tile's reason. Image inputs get the same treatment.

Any exception inside the added logic is logged and the stock Tesseract/layer
result is kept: the plugin fails open.

## Compatibility guard

`SUPPORTED_PAPERLESS` in `paperless_vlm_ocr/__init__.py` lists the Paperless
`(major, minor)` versions the plugin runs on, and `VERIFIED_PAPERLESS` right
above it names the image CI verifies the plugin against. On any other minor
`score()` returns `None`, so ingestion falls back to stock Tesseract instead of
breaking.

The parser scores `15`, above the built-in Tesseract parser (`10`), and is only
selected when `PAPERLESS_VLM_OCR_ENABLED=true` and endpoint, API key and model
are all set.

## Installation

The published image is a data-only OCI image (built `FROM scratch`): it contains
exactly the plugin tree at the image root (`paperless_vlm_ocr/` plus
`paperless_vlm_ocr-<version>.dist-info/`). Mount it read-only on a directory that
is on `PYTHONPATH`; `importlib.metadata` discovers the `dist-info` directory
there and Paperless finds the `paperless_ngx.parsers` entry point.

Requires Kubernetes 1.33+ for the image variant (app-template 5.3.0 gates the
`image` persistence type on it, and image volumes are read-only by design), and
a Paperless image that already ships `openai`, Pillow, `pdftotext`, `pdftoppm`
and `tesseract` (the official image does).

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

The image is not runnable; bind-mount the `site/` tree instead:

```sh
docker run --rm \
  -v "$PWD/site:/usr/src/paperless/vlm-ocr:ro" \
  -e PYTHONPATH=/usr/src/paperless/vlm-ocr \
  -e PAPERLESS_VLM_OCR_ENABLED=true \
  -e PAPERLESS_VLM_OCR_ENDPOINT=https://example.invalid/v1 \
  -e PAPERLESS_VLM_OCR_API_KEY=... \
  -e PAPERLESS_VLM_OCR_MODEL=... \
  ghcr.io/paperless-ngx/paperless-ngx:3.5.1
```

Check discovery with:

```sh
docker run --rm --read-only --tmpfs /tmp \
  -v "$PWD/site:/usr/src/paperless/vlm-ocr:ro" \
  -e PYTHONPATH=/usr/src/paperless/vlm-ocr \
  --entrypoint python3 ghcr.io/paperless-ngx/paperless-ngx:3.5.1 \
  -c "from importlib.metadata import entry_points; print([e.value for e in entry_points(group='paperless_ngx.parsers')])"
```

## Configuration

All settings come from the environment and are read on every parse.

| Variable | Default | Meaning |
|---|---|---|
| `PAPERLESS_VLM_OCR_ENABLED` | `false` | `true`/`1` enables the plugin |
| `PAPERLESS_VLM_OCR_ENDPOINT` | *(empty)* | OpenAI-compatible base URL, ending in `/v1` |
| `PAPERLESS_VLM_OCR_API_KEY` | *(empty)* | API key for that endpoint |
| `PAPERLESS_VLM_OCR_MODEL` | *(empty)* | Vision model name |
| `PAPERLESS_VLM_OCR_GATE_MODE` | `shadow` | `shadow` or `enforce` (anything else is `shadow`) |
| `PAPERLESS_VLM_OCR_GATE_THRESHOLD` | `0.9` | Agreement threshold for `coverage`/`support` |
| `PAPERLESS_VLM_OCR_GATE_MIN_TOKENS` | `15` | Minimum tokens for a layer/reference to be judged |
| `PAPERLESS_VLM_OCR_TIMEOUT_S` | `120` | Request timeout in seconds |
| `PAPERLESS_VLM_OCR_DOC_BUDGET_S` | `900` | Wall-clock budget for one parse, counted from the start of `parse()` (seconds) |
| `PAPERLESS_VLM_OCR_CONCURRENCY` | `4` | Parallel model requests per document |
| `PAPERLESS_VLM_OCR_MAX_PAGES` | `50` | Documents with more pages are left untouched |
| `PAPERLESS_VLM_OCR_IMAGE_PX` | `1800` | Long side of each page/tile image sent to the model |
| `PAPERLESS_VLM_OCR_DECISION_LOG` | `<DATA_DIR>/vlm-ocr/decisions.jsonl` | JSONL decision log path |

## Logging

The plugin logs on `paperless.parsing.vlm_ocr`, so its lines land in
`paperless.log` (with the rest of `paperless.*`) and on the worker's stdout. The
Tesseract parser keeps its own `paperless.parsing.tesseract` logger, so stock
messages such as `pdftotext exited 0` are unchanged.

One start line per document, one line per page, one line per model call and one
summary per document (`file` is the name only, `sha` its first 12 hex digits):

```
vlm_ocr start file=statement.pdf sha=a1b2c3d4e5f6 mime=application/pdf pages=5 archive=yes gate_mode=shadow threshold=0.90 model=vision-model
vlm_ocr page 2/5 sha=a1b2c3d4e5f6 kind=gated decision=keep coverage=0.987 support=0.991 threshold=0.90 layer_tokens=120 ref_tokens=118 source=layer reason=- tiles=1 loop_at=-
vlm_ocr page 1/5 sha=a1b2c3d4e5f6 kind=empty_layer decision=vlm coverage=- support=- threshold=0.90 layer_tokens=0 ref_tokens=- source=vlm reason=- vlm_vs_fallback_coverage=0.941 vlm_vs_fallback_support=0.933 tiles=1 loop_at=-
vlm_ocr page 4/5 sha=a1b2c3d4e5f6 kind=gated decision=vlm coverage=0.312 support=0.298 threshold=0.90 layer_tokens=87 ref_tokens=95 source=tesseract reason=RuntimeError tiles=1 loop_at=-
vlm_ocr page 3/5 sha=a1b2c3d4e5f6 kind=empty_layer decision=vlm coverage=- support=- threshold=0.90 layer_tokens=0 ref_tokens=- source=vlm reason=loop_cut vlm_vs_fallback_coverage=0.812 vlm_vs_fallback_support=0.774 tiles=3 loop_at=533
vlm_ocr llm page=1 sha=a1b2c3d4e5f6 model=vision-model host=api.example.test attempts=2 finish_reason=stop tokens_in=1700 tokens_out=420 seconds=12.3 image=1800x2546 bytes=731003 chars=47 response_id=chatcmpl-abc request_id=req-123 tiles=1 reason=- loop_at=-
vlm_ocr llm page=3 sha=a1b2c3d4e5f6 model=vision-model host=api.example.test failed error=APIStatusError status=429 request_id=req-999 seconds=3.5 tiles=1
vlm_ocr llm page=2 sha=a1b2c3d4e5f6 model=vision-model host=api.example.test attempts=2 finish_reason=stop tokens_in=1700 tokens_out=420 seconds=12.3 image=1800x2546 bytes=731003 chars=167 response_id=chatcmpl-abc request_id=req-123 rejected reason=repetition fallback_chars=812 tiles=1 loop_at=-
vlm_ocr llm page=3 sha=a1b2c3d4e5f6 model=vision-model host=api.example.test attempts=3 finish_reason=length tokens_in=2600 tokens_out=640 seconds=18.7 image=1042x1802 bytes=642110 chars=1240 response_id=chatcmpl-def request_id=req-456 tiles=3 reason=loop_cut loop_at=533
vlm_ocr summary file=statement.pdf sha=a1b2c3d4e5f6 pages=5 keep=3 vlm=1 fallback=1 would_vlm=0 no_signal=0 gate_mode=shadow threshold=0.90 coverage_min=0.312 coverage_mean=0.649 support_min=0.298 support_mean=0.644 sources=LLVTL content=assembled tokens_in=4300 tokens_out=1060 seconds=18.0 budget_exhausted=no
```

- `coverage`/`support` are `-` when the page has no reference read; `reason` is
  the final reason: `loop_cut` when a prefix was kept, otherwise a rejection or
  fallback reason, or the exception class of a failed attempt.
- `tiles` is the number of tiles the page was split into (1 for a normal page);
  `loop_at` is the offset of the detected loop inside that call's text (`-` when
  none).
- `vlm_vs_fallback_*` compares the model text with the text it displaced, so a
  suspicious `source=vlm` page is visible without opening the document.
- The summary's `sources` has one letter per page (`L` layer, `V` model,
  `T` Tesseract) and `content` says whether `document.text` was rewritten
  (`assembled`) or left as the parent parser produced it (`parent`);
  `budget_exhausted=yes` means at least one page stopped on the document
  budget (its reason is `budget`).
- Lines never contain the API key, the endpoint path or query, request bodies or
  transcribed text: failures report the exception class, HTTP status and request
  id, while calls - accepted or rejected - report attempts, sizes, token counts
  and the response/request ids only.

## Decision log

One JSON line per page, appended whether or not anything changed:

`ts`, `sha256` (of the parsed file), `mime`, `file`, `page`, `pages`, `kind`
(`gated`/`empty_layer`), `decision` (`keep`/`vlm`/`would_vlm`/`no_signal`),
`coverage`, `support`, `threshold`, `layer_tokens`, `ref_tokens`, `source`
(`layer`/`vlm`/`tesseract`), `fallback_reason`, `vlm_seconds`, `input_tokens`,
`output_tokens`, `model`, `finish_reason`, `attempts`, `response_id`,
`request_id`, `payload_bytes`, `image_px`, `vlm_chars`, `fallback_chars`,
`layer_chars`, `vlm_vs_fallback_coverage`, `vlm_vs_fallback_support`,
`error_status`, `gate_mode`, `plugin_version`, `tiles`, `loop_at`, `reason`.

`fallback_reason` carries a rejection reason (including `budget` when the
document budget stopped the page), a fallback failure, or the exception class of
a failed model call (with its HTTP status in `error_status`);
`reason` is the final outcome (`loop_cut` for an accepted prefix, `budget` when
the document budget stopped the page, otherwise the same value as
`fallback_reason`). `tiles` and `loop_at` mirror the log line;
`attempts` counts the first request plus retries (including a second call for a
rejected page), and `response_id`/`request_id` identify the last call upstream.
A failing log write is a warning, never a parse failure. Use the `sha256` to map
a record back with `Document.objects.filter(checksum=...)`.

## Offline gate scan

`scan.py` evaluates the gate on PDFs without Django and without the vision
model, always in enforce mode, so you can review which pages would be routed:

```sh
PYTHONPATH=site python3 -m paperless_vlm_ocr.scan \
  --lang deu+eng --jobs 8 --threshold 0.9 --min-tokens 15 \
  --out results.jsonl *.pdf
```

It writes one record per page (`file`, `page`, `layer_tokens`, `ref_tokens`,
`coverage`, `support`, `decision`), where `decision` is
`keep|vlm|no_signal|empty_layer`, plus `error` records for unreadable files. It
needs `pdftotext`, `pdftoppm` and `tesseract` on `PATH`.

## Releasing

The version lives in three places and they must move together:

1. `__version__` in `site/paperless_vlm_ocr/__init__.py`,
2. the dist-info directory name `site/paperless_vlm_ocr-<version>.dist-info/`,
3. the `Version:` line in that directory's `METADATA`.

Bump all three in one pull request and merge it to `main`. CI verifies the
plugin against the Paperless-ngx image named by `VERIFIED_PAPERLESS`, runs the
unit tests and a real parse on a one-page scan inside that image, and then:

1. publishes `ghcr.io/anokfireball/paperless-vlm-ocr:<version>` for
   `linux/amd64` and `linux/arm64`,
2. creates the `v<version>` tag and a GitHub release with generated notes.

A push to `main` whose version is already tagged publishes nothing: no image is
rebuilt and no release is recreated.

Renovate keeps `VERIFIED_PAPERLESS` current; the annotation above it in
`site/paperless_vlm_ocr/__init__.py` is a `renovate.json` custom manager. Its
post-upgrade task, `node .github/scripts/renovate-paperless-bump.mjs`, either

- leaves everything alone for a patch update of an already supported minor
  (the CI in-image checks, entry point and smoke parse, prove the new image on
  the PR), or
- adds the new `(major, minor)` to `SUPPORTED_PAPERLESS` and bumps the plugin's
  patch version in all three places, so merging the PR publishes a new image.

`site/` must not contain tab characters; CI fails on them. `$` is fine.

### Compatibility and upgrade order

On a Paperless minor that is not in `SUPPORTED_PAPERLESS`, `score()` returns
`None`: the plugin steps aside with a warning and ingestion uses stock Tesseract
parsing. A Paperless upgrade therefore cannot break parsing, even before a
plugin release catches up.

When upgrading Paperless to a new minor, merge and deploy the plugin release
that lists it first. Listing a new minor keeps the older ones (they stay in
`SUPPORTED_PAPERLESS`), so that release still runs on the Paperless version you
are replacing, and deploying it before the Paperless upgrade leaves no window
without the plugin.
