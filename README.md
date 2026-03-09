# Etsytool

Private internal app for one Etsy shop to bulk create/update listings, manage variations, upload images, and list existing products.

## Requirements

- Python 3.9+
- Etsy developer app credentials
- PDM (`pdm`)

Install dependencies:

```bash
pdm install
```

## Config

Use `config.json` with at least:

- `client_id`
- `client_secret` (optional but supported)
- `shop_id`
- `redirect_uri`
- `auth_url`
- `token_url`

## Commands

Validate upload input (no API writes):

```bash
pdm run python etsy_bulk_uploader.py --config config.json --input products.json --images ./images --dry-run
```

Upload listings + images:

```bash
pdm run python etsy_bulk_uploader.py --config config.json --input products.json --images ./images
```

List products/listings from Etsy:

```bash
pdm run python etsy_bulk_uploader.py --config config.json --list-products --state active --limit 25 --max-pages 2
```