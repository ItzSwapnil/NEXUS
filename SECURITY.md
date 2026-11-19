# Security policy

NEXUS connects to external broker services and can handle credentials, cookies, account data, and local research artifacts. Treat local configuration and runtime output as sensitive.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through [GitHub’s security advisory form](https://github.com/ItzSwapnil/NEXUS/security/advisories/new) rather than posting credentials, exploit details, or account data in a public issue. If private reporting is unavailable for this repository, open a minimal issue asking the maintainer for a private contact method; do not include the vulnerability details in that issue.

Include the affected version or commit, impact, and concise reproduction steps. Redact tokens, cookies, account identifiers, and personal data.

## Supported versions

Security fixes are currently prepared for the latest state of the default branch. There is no published support window for older releases.

## Protect your environment

- Keep `.env`, browser sessions, cookies, private keys, logs, reports, and broker data out of source control and public storage.
- Keep `.env.example` limited to placeholders.
- Use demo mode while evaluating changes, and remember broker-backed demo workflows still contact the broker.
- If a credential or token is exposed, revoke or rotate it promptly. Removing a file does not remove it from Git history.
- Review code and network behavior before connecting an account or exposing the web API.

## Project limitations

NEXUS is experimental. It does not provide a security guarantee or formal security audit. Do not expose a broker-connected service publicly without reviewing authentication, network access, and deployment configuration.
