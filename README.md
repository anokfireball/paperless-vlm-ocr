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

Model output is rejected (and the fallback used) when it is truncated
(`finish_reason=length`), empty, repetitive, or a runaway expansion. Any
exception inside the added logic is logged and the stock Tesseract/layer result
is kept: the plugin fails open.

## Compatibility guard

`SUPPORTED_PAPERLESS` in `paperless_vlm_ocr/__init__.py` lists the verified
Paperless `(major, minor)` versions. On any other version `score()` returns
`None`, so ingestion falls back to stock Tesseract instead of breaking.

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
    image: ghcr.io/anokfireball/paperless-vlm-ocr:0.1.1
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
| `PAPERLESS_VLM_OCR_CONCURRENCY` | `4` | Parallel model requests per document |
| `PAPERLESS_VLM_OCR_MAX_PAGES` | `50` | Documents with more pages are left untouched |
| `PAPERLESS_VLM_OCR_IMAGE_PX` | `1800` | Long side of the page image sent to the model |
| `PAPERLESS_VLM_OCR_DECISION_LOG` | `<DATA_DIR>/vlm-ocr/decisions.jsonl` | JSONL decision log path |

## Decision log

One JSON line per page, appended whether or not anything changed:

`ts`, `sha256` (of the parsed file), `mime`, `page`, `pages`, `kind`
(`gated`/`empty_layer`), `decision` (`keep`/`vlm`/`would_vlm`/`no_signal`),
`coverage`, `support`, `layer_tokens`, `ref_tokens`, `source`
(`layer`/`vlm`/`tesseract`), `fallback_reason`, `vlm_seconds`, `input_tokens`,
`output_tokens`, `gate_mode`, `plugin_version`.

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

Then tag `vX.Y.Z` (without a `v` inside the version) and push the tag. CI
verifies the tag against `__version__`, reruns the tests and publishes
`ghcr.io/anokfireball/paperless-vlm-ocr:X.Y.Z` for `linux/amd64` and
`linux/arm64`.

Supporting a new Paperless minor release: verify the plugin against the new
image, add the `(major, minor)` tuple to `SUPPORTED_PAPERLESS`, bump the version,
and update `PAPERLESS_IMAGE` in `.github/workflows/ci.yaml`.

`site/` must not contain tab characters; CI fails on them. `$` is fine.
