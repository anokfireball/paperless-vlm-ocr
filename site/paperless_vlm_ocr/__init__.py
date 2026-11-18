"""Paperless-ngx parser plugin: per-page text-layer gate plus VLM transcription."""

__version__ = "0.2.0"

# renovate: datasource=docker depName=ghcr.io/paperless-ngx/paperless-ngx
VERIFIED_PAPERLESS = "3.5.1"

# Paperless (major, minor) versions the plugin runs on; older minors stay listed.
SUPPORTED_PAPERLESS = {(3, 5)}
