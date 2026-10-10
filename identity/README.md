# M4: optional local identity

Implemented as an optional local CLI, disabled unless `--enable-local-identity` is supplied. Explicit consent is required for enrollment; derived templates and consent can be deleted together. No gateway or cloud integration exists. Read `python -m identity.cli consent` and [the M4 guide](../docs/M4_RESEARCH.md) before use. Local SQLite is plaintext; use an unsynced directory. No real person was enrolled during implementation.
