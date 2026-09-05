# Security Policy

## Scope

AI Meets Control Theory (`aimct`) is a research and educational codebase:
from-scratch control, estimation, and reinforcement-learning implementations,
benchmark experiments, and interactive visualization sandboxes. It has no
network service, no authentication, and does not process user data — its
attack surface is the library and CLI running locally on a developer's or
student's own machine.

Realistic concerns worth reporting: a dependency with a known CVE pinned in
`pyproject.toml`, an unsafe deserialization path (e.g. `pickle`/`eval` on
untrusted input), or a bug that could crash or hang a real-time control loop
in a way that would matter if this code were ever driving physical hardware.

## Supported Versions

Only the latest release on the `main` branch is supported. This project is
pre-1.0 (see `CHANGELOG.md`); there is no long-term-support branch.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for a security concern. Instead,
use GitHub's private reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** to open a private advisory.

This reaches the maintainers directly without disclosing the issue publicly
before a fix is available. We'll acknowledge reports as promptly as we can
and credit the reporter in the fix (unless you'd prefer otherwise).
