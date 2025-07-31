# 🛡️ NEXUS Security Policy

<p align="center">
  <img src="https://img.shields.io/badge/security-critical-red" alt="Security Critical" />
  <img src="https://img.shields.io/badge/vulnerability-disclosure-blue" alt="Vulnerability Disclosure" />
  <img src="https://img.shields.io/badge/patches-welcome-brightgreen" alt="Patches Welcome" />
  <img src="https://img.shields.io/badge/encryption-recommended-blueviolet" alt="Encryption" />
</p>

---

## Supported Versions

| Version | Supported          |
| ------- | ----------------- |
| latest  | :white_check_mark:|

We support the latest major version of NEXUS. Please update to the latest release before reporting security issues.

---

## 📣 Reporting a Vulnerability

If you discover a security vulnerability, **do not open a public issue**. Instead, report it confidentially by emailing **security@nexus.ai**.

Please include:
- A detailed description of the vulnerability
- Steps to reproduce (code, screenshots, logs, etc.)
- Potential impact and affected components
- Your contact information (for follow-up)

We will acknowledge your report within 48 hours and provide updates as we investigate and address the issue. You will be kept informed throughout the process.

---

## 🔒 Security Best Practices

- Always keep your dependencies up to date (use `uv pip install -U ...`)
- Use strong authentication for all integrations and API keys
- Never share your API keys, credentials, or sensitive data
- Regularly review your logs and monitor for suspicious activity
- Use encrypted storage for sensitive data (see [README.md](../README.md#secure-data-handling))
- Follow the principle of least privilege for all access
- Use 2FA on all accounts where possible

---

## 🕵️‍♂️ Disclosure Policy

We will coordinate with you to validate and address the vulnerability before public disclosure. Credit will be given to reporters unless requested otherwise. We follow responsible disclosure best practices.

---

## 🙏 Thank You

We appreciate your efforts to make NEXUS more secure! Security is a community effort—your vigilance helps protect everyone.

---

> _If you have questions about security, responsible disclosure, or best practices, please contact the maintainers at **security@nexus.ai**._
