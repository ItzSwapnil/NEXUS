# NEXUS Security Policy

Security policy for the NEXUS autonomous trading system.

## Supported Version
Latest `main` branch only.

## Reporting Vulnerabilities
Report security vulnerabilities via GitHub Issues (mark as security-related). Include:
- Description & impact
- Reproduction steps / PoC
- Affected modules

## Current Security Posture
| Area | Status | Notes |
|------|--------|-------|
| Credential storage | Plaintext | .env file (local only) |
| Network calls | Active | Broker API integration |
| Logging | Plaintext | Standard logging |
| Encryption | None | Future enhancement |
| Secrets in repo | Not intended | Use `.env` (gitignored); keep `.env.example` placeholder-only |

## Best Practices
- Keep `.env` file secure and never commit it
- Never force-add ignored files containing credentials, tokens, private keys, or broker data
- If a secret was committed, revoke or rotate it immediately; deleting the file does not remove it from Git history
- Use demo mode for testing
- Sanitize logs, reports, and session files before sharing
- Review code before trading with real funds
- Formal threat modeling
- Penetration testing
- Secure secret management
- Encrypted persistence layers

## Disclaimer
Security guarantees are **not** provided. Use at your own risk.
