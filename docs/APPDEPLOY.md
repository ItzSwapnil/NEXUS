# Remote access and deployment notes

NEXUS is a Python ASGI application. The repository includes a `Dockerfile` and a helper for starting a Cloudflare Tunnel. These are deployment examples; review them and your hosting provider’s current configuration before exposing a service.

## Local container

Build and start the container from the repository root:

```bash
docker build -t nexus-dashboard .
docker run --rm -p 8000:8000 -e PORT=8000 nexus-dashboard
```

The image starts the web dashboard in demo mode, sets `AUTO_LOGIN=false`, and disables real web trading through `NEXUS_ALLOW_REAL_WEB_TRADING=false`. Demo mode may still access configured external services. Do not add broker credentials to the image or build context.

## Protected local API tunnel

Before exposing a local API, generate a dedicated service token. Copy it into a local secret store so the same value can be supplied to both terminal sessions. Never put it in frontend code or a public URL.

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
export NEXUS_API_TOKEN='<paste the generated token>'
export NEXUS_API_REQUIRE_TOKEN=true
uv run python -m nexus.main --web --demo --host 127.0.0.1 --port 8000
```

In another terminal, run:

```bash
export NEXUS_API_TOKEN='<the same token configured for the NEXUS process>'
export NEXUS_API_REQUIRE_TOKEN=true
bash scripts/start_appdeploy_tunnel.sh
```

The helper requires `NEXUS_API_REQUIRE_TOKEN=true`. Without a configured named tunnel it starts a temporary Cloudflare Tunnel. Temporary tunnel URLs can change and are intended for development use.

For a named Cloudflare Tunnel, configure and route the tunnel with Cloudflare first, then set its local name:

```bash
export NEXUS_CLOUDFLARE_TUNNEL=nexus-api
bash scripts/start_appdeploy_tunnel.sh
```

The local NEXUS host must remain running and reachable. Store the public API base URL and service token using the consuming service’s secret-management feature. Rotate the token if it is exposed.

## Exposure boundaries

- Do not expose the Quotex login flow, credentials, cookies, or broker controls to an untrusted public frontend.
- Keep real web trading disabled unless a separate operational review explicitly requires it.
- Use a dedicated token and TLS-protected tunnel for remote API access.
- Review API authentication and allowed operations in `nexus/web/` before deployment; this document is not a security audit.
