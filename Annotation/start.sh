#!/usr/bin/env bash
# Main entry point — starts the web annotator.
exec "$(cd "$(dirname "$0")" && pwd)/start_web_annotator.sh" "$@"
