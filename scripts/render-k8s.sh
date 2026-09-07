#!/bin/bash
# Render Kubernetes manifests using envsubst.
# Requires: apt install gettext-base  (or brew install gettext on macOS)
#
# Usage:
#   cp .env.prod .env   # fill in real values
#   ./scripts/render-k8s.sh
#   kubectl apply -f deploy/kubernetes/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$PROJECT_ROOT/deploy/kubernetes/deployment.yaml"
OUTPUT_DIR="${KUSTOMIZE_OUTPUT_DIR:-$PROJECT_ROOT/deploy/kubernetes/dist}"

if ! command -v envsubst &> /dev/null; then
    echo "envsubst not found. Install gettext-base (Debian/Ubuntu) or gettext (macOS: brew install gettext)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
envsubst < "$MANIFEST" > "$OUTPUT_DIR/deployment.yaml"
echo "Rendered: $OUTPUT_DIR/deployment.yaml"
