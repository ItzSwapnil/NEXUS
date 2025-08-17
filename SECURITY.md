# NEXUS Security Policy (Current Minimal Scope)

This project is an early-stage prototype. No production credential handling or encryption logic is implemented yet. Treat it as **research-only** software.

## Supported Version
Latest `main` branch only.

## Reporting Vulnerabilities
Email: security@nexus.ai (placeholder) or open a private advisory on GitHub if available. Please include:
- Description & impact
- Reproduction steps / PoC
- Affected modules

We will acknowledge legitimate reports as soon as feasible; response times are best-effort.

## Current Security Posture
| Area | Status | Notes |
|------|--------|-------|
| Credential storage | Not implemented | Config holds values in plaintext YAML (local) |
| Network calls | Not active | Live pyquotex integration not wired yet |
| Logging | Plaintext | Override audit only |
| Encryption | None | Future roadmap item |
| Secrets in repo | Should be none | Do not commit real credentials |

## Best Practices (User)
- Keep repository private if experimenting with real credentials.
- Do **not** trade live funds with this alpha.
- Sanitize logs before sharing.
- Remove or encrypt `config.yaml` if adding credentials.

## Out of Scope (For Now)
- Formal threat modeling
- Penetration testing
- Secure secret management
- Encrypted persistence layers

## Disclaimer
Security guarantees are **not** provided. Use at your own risk.
