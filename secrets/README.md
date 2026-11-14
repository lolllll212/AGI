# secrets/

Nothing in this folder is committed except this file (see `.gitignore`).

| File | What it is | Used by |
| --- | --- | --- |
| `agi_api_token.txt` | The bearer token for the API. At least 16 characters. | the `api` container (`--token-file`), Prometheus (to scrape `/metrics`) |
| `keys.json` | Optional: one key per caller, `{"alice": "<secret>", "ci": "<secret>"}` | `--keys-file` |

Create the token:

    python -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/agi_api_token.txt

TLS certificates for the reverse proxy go in `monitoring/nginx/ssl/` (`fullchain.pem`, `privkey.pem`).
