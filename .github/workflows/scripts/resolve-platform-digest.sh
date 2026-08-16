#!/usr/bin/env bash
# Emit the PLATFORM manifest digest for one pushed image ref so deploy can
# pin the exact digest the head's crane resolves (see
# src/head/internal/registry/manager.go verifyImageDigest).
#
# WHY the platform digest, and NOT `podman image inspect --format
# '{{index .RepoDigests 0}}'`:
# - Single-arch image (our engine images): RepoDigests[0] IS the platform
#   manifest digest — matches. skopeo inspect .Digest agrees.
# - Manifest LIST (multi-arch) image: RepoDigests[0] is the MANIFEST-LIST
#   digest, but the head's crane resolves the PLATFORM manifest digest for
#   the pushed arch. Pinning the list digest produces the deploy-time
#   "image digest mismatch: got <platform>, expected <list>" error and a
#   redeploy cycle. This script always emits the PLATFORM digest, so
#   deploy scripts can pin it directly.
#
# How the digest is resolved:
# - skopeo inspect --raw docker://<ref> returns the raw manifest. An OCI
#   index (manifest list) carries a .manifests array; a single-arch OCI
#   manifest does not. For a list we select the linux/amd64 platform
#   manifest (both sm86-sm120 and sm60 targets are x86_64). For a
#   single-arch manifest, skopeo inspect --format '{{.Digest}}' returns
#   the manifest digest, which IS the platform digest.
# - Fallback (no skopeo): the freshly-pushed local image's RepoDigests[0].
#   Only valid for the image pushed in THIS job (single-arch push => the
#   platform digest), and only because the local copy is brand new; skopeo
#   is the registry-authoritative path.
#
# Usage: resolve-platform-digest.sh <repo>:<tag> <output-var-name> "<label>"
#   <output-var-name>=sha256:<platform-manifest-digest>  -> $GITHUB_OUTPUT
#   "PLATFORM_DIGEST=sha256:<...>" line + pin table       -> $GITHUB_STEP_SUMMARY
set -euo pipefail

REF="$1"
VAR_NAME="$2"
LABEL="${3:-image}"

DIGEST=""
if command -v skopeo >/dev/null 2>&1; then
  RAW=$(skopeo inspect --raw "docker://${REF}" 2>/dev/null || true)
  if [ -n "$RAW" ] && echo "$RAW" | jq -e '.manifests != null' >/dev/null 2>&1; then
    # Manifest list: platform manifest digest for the pushed arch.
    DIGEST=$(echo "$RAW" | jq -r '.manifests[] | select(.platform.os == "linux" and .platform.architecture == "amd64") | .digest' | head -n1)
  else
    # Single-arch manifest: its digest IS the platform digest.
    DIGEST=$(skopeo inspect --format '{{.Digest}}' "docker://${REF}" 2>/dev/null || true)
  fi
else
  DIGEST=$(podman image inspect "${REF}" --format '{{index .RepoDigests 0}}' 2>/dev/null | sed 's|.*@||' || true)
fi

if [ -z "$DIGEST" ]; then
  echo "::error::Could not resolve platform manifest digest for ${REF}" >&2
  exit 1
fi

echo "${VAR_NAME}=${DIGEST}" >> "$GITHUB_OUTPUT"

{
  echo "### Deploy pin — ${LABEL}"
  echo ""
  echo "| Field | Value |"
  echo "|-------|-------|"
  echo "| Image | \`${REF}\` |"
  echo "| Platform manifest digest | \`${DIGEST}\` |"
  echo ""
  echo "\`PLATFORM_DIGEST=${DIGEST}\`"
  echo ""
  echo "Pin the head's \`-llama-image-digest\` to \`${DIGEST}\` — the platform manifest digest (NOT \`RepoDigests[0]\`, which is the manifest-LIST digest for multi-arch images)."
} >> "$GITHUB_STEP_SUMMARY"

echo "PLATFORM_DIGEST=${DIGEST} (${REF})"
