# Device Identity And Auth Workflow

This document defines the minimum identity contract between the ESP device and the server.

## Identity Tuple

Each device is identified by a 2-tuple:

- `device_id`
- `device_secret`

The tuple is provisioned and validated as follows:

- `device_id` is public and stable.
- `device_secret` is private and long-lived.
- The server should index devices by `device_id` and then verify `device_secret`.
- The server should not log or expose `device_secret` in plaintext.

Operationally, the tuple is the credential. In storage, the server should treat `device_id` as the lookup key and the `device_secret` as the verifier material.

## Device ID

`device_id` is derived from the ESP factory/base MAC address burned into eFuse.

- Source: factory/base MAC from ESP-IDF, not a soft-configured Wi-Fi MAC.
- Raw size: 6 bytes.
- Canonical wire format: 12-character lowercase hexadecimal string.
- Example: `34cdab12ef90`
- Allowed characters: `0-9`, `a-f`
- Length: exactly 12 ASCII characters
- Mutability: immutable for the lifetime of the board

Why this format:

- 48 bits is enough for board-level uniqueness when sourced from the factory MAC.
- Lowercase hex avoids delimiter ambiguity and is easy to use in URLs, logs, and database keys.
- The identifier remains compact and deterministic.

## Device Secret

`device_secret` is generated once on first boot and then stored in NVS.

- Source: ESP hardware RNG via `esp_fill_random`
- Raw size: 32 bytes
- Stored format on device: raw NVS blob
- Recommended external/provisioning format: 64-character lowercase hexadecimal string
- Rotation: only through an explicit reprovisioning flow
- Logging: never log the full value

Why 32 bytes:

- 256 bits is well above the brute-force margin needed for device authentication.
- It is suitable for direct bearer comparison or HMAC-based request signing.
- It avoids weak MAC-derived secrets, which would be predictable.

## Server-Side Requirements

- Enforce uniqueness on `device_id`.
- Treat `device_secret` as credential material, not metadata.
- If the server uses HMAC request signing, it must retain access to the raw secret or a reversible encrypted form.
- If the server only performs equality checks over TLS, it may store a memory-hard hash instead, but then it cannot later verify HMAC signatures with that same record.
- Compare secrets or MAC tags in constant time.
- Never use `device_id` alone as proof of device identity.

## ESP Boot Workflow

1. Read the factory/base MAC from eFuse.
2. Convert the 6-byte MAC into the canonical 12-char lowercase hex `device_id`.
3. Open NVS namespace `identity`.
4. Load `device_secret` from key `device_secret`.
5. If missing, generate 32 random bytes, persist them to NVS, and use that as the secret.
6. Keep the secret in memory only as long as needed for authentication.

## Recommended ESP <-> Server Auth Workflow

Use the identity tuple to establish a short-lived session instead of sending the raw secret on every data message.

### 1. Provisioning

- The server records `device_id`.
- The first known `device_secret` is registered through manufacturing, a secure local bootstrap flow, or an operator-only enrollment endpoint.

### 2. Session Authentication

The device opens a TLS connection and sends:

- `device_id`
- `timestamp` or monotonic counter
- `nonce`
- `signature`

Where:

- `signature = HMAC-SHA256(device_secret, canonical_auth_message)`

Recommended canonical auth message fields:

- protocol version
- `device_id`
- `timestamp`
- `nonce`
- optional firmware version

### 3. Server Verification

- Look up the device record by `device_id`.
- Recompute the HMAC with the stored device secret.
- Reject stale timestamps, reused nonces, or invalid signatures.
- Return a short-lived session token or mark the TLS connection as authenticated.

### 4. Telemetry/Data Phase

After auth succeeds:

- send telemetry using the authenticated session
- rotate session tokens frequently
- do not resend the raw secret in normal telemetry payloads

## Local Storage Contract

On the ESP, the `identity` namespace should own:

- key: `device_secret`
- type: blob
- size: exactly 32 bytes

Any other size should be treated as corrupted identity state and rejected rather than silently truncated.
