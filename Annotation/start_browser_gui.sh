#!/usr/bin/env bash
# DEPRECATED: slow noVNC desktop streaming. Use ./start.sh instead.
echo "NOTE: The noVNC desktop GUI is slow. Use the fast web annotator instead:"
echo "  ./start.sh"
echo ""
exec "$(cd "$(dirname "$0")" && pwd)/start_web_annotator.sh"
