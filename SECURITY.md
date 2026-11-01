# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Simulusion, please report it privately by opening a security advisory on GitHub:

https://github.com/paramthakkar123/simulusion/security/advisories/new

Please do **not** report security vulnerabilities through public GitHub issues, discussions, or pull requests.

## What to Include

- A clear description of the vulnerability
- Steps to reproduce (if applicable)
- Affected versions
- Any potential mitigations you have identified

## Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 5 business days
- **Fix and release**: timeline depends on severity, typically 7–30 days

## Scope

Every `torch.load` in Simulusion, including the `simulusion eval` / `simulusion play` entry points, the scripts in `scripts/` and the demos in `demos/`, passes `weights_only=True`, and subprocesses run with `subprocess.run(shell=False)`. Only open checkpoints from sources you trust all the same. If you find any code path that deviates from these patterns, please report it.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |
| 0.4.x   | Yes       |
| < 0.4   | No        |
