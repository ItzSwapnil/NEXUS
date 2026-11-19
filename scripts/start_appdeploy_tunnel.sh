#!/usr/bin/env bash
set -euo pipefail

origin_url="${NEXUS_TUNNEL_ORIGIN:-http://127.0.0.1:8000}"

if [[ -z "${NEXUS_API_TOKEN:-}" ]]; then
    echo "Set NEXUS_API_TOKEN before exposing NEXUS through a tunnel." >&2
    exit 1
fi

if [[ "${NEXUS_API_REQUIRE_TOKEN:-}" != "true" ]]; then
    echo "Set NEXUS_API_REQUIRE_TOKEN=true before starting the tunnel." >&2
    exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared is required. Install it before starting an AppDeploy tunnel." >&2
    exit 1
fi

if ! curl --fail --silent --show-error \
    -H "Authorization: Bearer ${NEXUS_API_TOKEN}" \
    "${origin_url}/api/status" >/dev/null; then
    echo "NEXUS is not reachable at ${origin_url} with the configured service token." >&2
    exit 1
fi

if [[ -n "${NEXUS_CLOUDFLARE_TUNNEL:-}" ]]; then
    echo "Starting named Cloudflare Tunnel ${NEXUS_CLOUDFLARE_TUNNEL} for ${origin_url}."
    echo "Keep its configured hostname as NEXUS_API_BASE_URL in AppDeploy."
    exec cloudflared tunnel run "${NEXUS_CLOUDFLARE_TUNNEL}"
fi

echo "Starting a temporary tunnel for ${origin_url}."
echo "Use the generated HTTPS URL only for development; set NEXUS_CLOUDFLARE_TUNNEL for a stable hostname."
exec cloudflared tunnel --url "${origin_url}"
