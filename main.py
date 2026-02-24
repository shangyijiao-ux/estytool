#!/usr/bin/env python3
"""
Etsy Bulk Listing Uploader (single-file starter)

What it does (fits your needs):
- Import products from CSV or JSON
- Validate required fields
- Create listings as DRAFT (draft-first)
- Upload multiple images per listing in a fixed order
- Create/update variations (size/color/pattern) + per-variant SKU/qty/price (where supported)
- Rate-limit friendly + retries

IMPORTANT:
Etsy’s exact endpoints and “variations/inventory” schemas differ by API version and auth type.
So this file is written as a robust framework with:
- Correct structure, validation, batching, image ordering, retries
- Clear placeholders where you must align to Etsy’s current docs for:
  * OAuth (PKCE) / token exchange
  * Listing create/update endpoint paths
  * Inventory/variations payload shape

Usage (example):
  python etsy_bulk_uploader.py --config config.json --input products.json --images ./images --dry-run
  python etsy_bulk_uploader.py --config config.json --input products.csv --images ./images

Config JSON example:
{
  "client_id": "YOUR_ETSY_APP_KEYSTRING",
  "redirect_uri": "http://localhost:8080/callback",
  "scopes": ["listings_r", "listings_w", "shops_r"],  // adjust to your approved scopes
  "token_file": "etsy_tokens.json",
  "api_base": "https://api.etsy.com/v3/application",
  "shop_id": 12345678
}

Input JSON example (one product):
{
  "title": "Hand-knitted Teddy Bear",
  "description": "Materials... Care... Processing time...",
  "price": 45.00,
  "quantity": 10,
  "taxonomy_id": 123,              // optional
  "tags": ["knit", "toy", "handmade"],
  "materials": ["cotton yarn"],     // optional
  "who_made": "i_did",
  "when_made": "made_to_order",
  "is_supply": false,
  "shipping_profile_id": 111,      // optional
  "return_policy_id": 222,         // optional
  "variations": {
    "size": ["S", "M", "L"],
    "color": ["Cream", "Brown"],
    "pattern": ["Stars", "Plain"]
  },
  "variants": [
    {"size":"S","color":"Cream","pattern":"Stars","sku":"BEAR-S-CR-STAR","qty":2,"price":45.00},
    {"size":"M","color":"Cream","pattern":"Stars","sku":"BEAR-M-CR-STAR","qty":2,"price":48.00}
  ],
  "images": [
    "bear_main.jpg",
    "bear_detail1.jpg",
    "bear_scale.jpg"
  ]
}

CSV support:
- Provide columns: title, description, price, quantity, tags (comma), images (comma), etc.
- Variants can be provided via a separate JSON file or embedded as JSON in a column.

"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import http.server
import json
import mimetypes
import os
import random
import re
import sys
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests


# -----------------------------
# Utilities
# -----------------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def load_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Union[str, Path], obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def split_csv_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def backoff_sleep(attempt: int, base: float = 0.8, cap: float = 10.0) -> None:
    # Exponential backoff with jitter
    t = min(cap, base * (2 ** attempt)) * (0.7 + random.random() * 0.6)
    time.sleep(t)


def guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


# -----------------------------
# Data Models
# -----------------------------

@dataclass
class Variant:
    size: Optional[str] = None
    color: Optional[str] = None
    pattern: Optional[str] = None
    sku: Optional[str] = None
    qty: Optional[int] = None
    price: Optional[float] = None


@dataclass
class Product:
    title: str
    description: str
    price: float
    quantity: int
    tags: List[str]
    images: List[str]

    taxonomy_id: Optional[int] = None
    materials: Optional[List[str]] = None
    who_made: str = "i_did"
    when_made: str = "made_to_order"
    is_supply: bool = False

    shipping_profile_id: Optional[int] = None
    return_policy_id: Optional[int] = None

    # Variation definitions and concrete variants
    variations: Optional[Dict[str, List[str]]] = None
    variants: Optional[List[Variant]] = None


# -----------------------------
# Input Parsing
# -----------------------------

def parse_products_from_json(path: Path) -> List[Product]:
    raw = load_json(path)
    items = raw if isinstance(raw, list) else [raw]
    products: List[Product] = []
    for it in items:
        products.append(_product_from_dict(it))
    return products


def parse_products_from_csv(path: Path) -> List[Product]:
    # Minimal CSV parser without pandas (single-file).
    import csv
    products: List[Product] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Allow JSON in some fields (variants/variations)
            variations = None
            variants = None
            if row.get("variations"):
                try:
                    variations = json.loads(row["variations"])
                except Exception:
                    variations = None
            if row.get("variants"):
                try:
                    variants_raw = json.loads(row["variants"])
                    variants = [Variant(**vr) for vr in variants_raw]
                except Exception:
                    variants = None

            prod = Product(
                title=row.get("title", "").strip(),
                description=row.get("description", "").strip(),
                price=float(row.get("price", "0") or 0),
                quantity=int(float(row.get("quantity", "0") or 0)),
                tags=split_csv_list(row.get("tags", "")),
                images=split_csv_list(row.get("images", "")),
                taxonomy_id=int(row["taxonomy_id"]) if row.get("taxonomy_id") else None,
                materials=split_csv_list(row.get("materials", "")) or None,
                who_made=row.get("who_made", "i_did") or "i_did",
                when_made=row.get("when_made", "made_to_order") or "made_to_order",
                is_supply=str(row.get("is_supply", "false")).lower() in ("1", "true", "yes"),
                shipping_profile_id=int(row["shipping_profile_id"]) if row.get("shipping_profile_id") else None,
                return_policy_id=int(row["return_policy_id"]) if row.get("return_policy_id") else None,
                variations=variations,
                variants=variants,
            )
            products.append(prod)
    return products


def _product_from_dict(d: Dict[str, Any]) -> Product:
    variants = None
    if d.get("variants"):
        variants = [Variant(**v) for v in d["variants"]]
    return Product(
        title=str(d.get("title", "")).strip(),
        description=str(d.get("description", "")).strip(),
        price=float(d.get("price", 0)),
        quantity=int(d.get("quantity", 0)),
        tags=list(d.get("tags", []) or []),
        images=list(d.get("images", []) or []),
        taxonomy_id=d.get("taxonomy_id"),
        materials=d.get("materials"),
        who_made=d.get("who_made", "i_did"),
        when_made=d.get("when_made", "made_to_order"),
        is_supply=bool(d.get("is_supply", False)),
        shipping_profile_id=d.get("shipping_profile_id"),
        return_policy_id=d.get("return_policy_id"),
        variations=d.get("variations"),
        variants=variants,
    )


# -----------------------------
# Validation
# -----------------------------

def validate_product(p: Product) -> List[str]:
    errs: List[str] = []
    if not p.title or len(p.title) < 3:
        errs.append("title is required (min 3 chars)")
    if not p.description or len(p.description) < 10:
        errs.append("description is required (min 10 chars)")
    if p.price <= 0:
        errs.append("price must be > 0")
    if p.quantity < 0:
        errs.append("quantity must be >= 0")
    if not p.images:
        errs.append("at least 1 image is required")
    if any(not isinstance(t, str) or not t.strip() for t in p.tags):
        errs.append("tags must be non-empty strings")
    if p.variants:
        for i, v in enumerate(p.variants):
            if v.qty is not None and v.qty < 0:
                errs.append(f"variants[{i}].qty must be >= 0")
            if v.price is not None and v.price <= 0:
                errs.append(f"variants[{i}].price must be > 0")
    return errs


# -----------------------------
# OAuth (PKCE) helper
# -----------------------------
# NOTE: This is a generic PKCE pattern. You must align auth URLs/params with Etsy’s current docs.

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def make_pkce_pair() -> Tuple[str, str]:
    verifier = _b64url(os.urandom(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    # stores query params into server object
    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        self.server.query_params = qs  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"You can close this window now.")
    def log_message(self, format, *args):  # silence
        return


def oauth_authorize_interactive(
    *,
    client_id: str,
    redirect_uri: str,
    scopes: List[str],
    auth_url: str,
    token_url: str,
    api_key_header_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic OAuth PKCE interactive flow:
    - Opens browser
    - Starts local server to catch callback
    - Exchanges code for token

    You MUST set auth_url and token_url to Etsy's current endpoints.
    Some Etsy setups use x-api-key header or client_id in header; adjust accordingly.
    """
    verifier, challenge = make_pkce_pair()
    state = _b64url(os.urandom(12))

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    url = auth_url + "?" + urllib.parse.urlencode(params)
    eprint("Opening authorization URL in browser...")
    webbrowser.open(url)

    # start local callback server
    parsed = urllib.parse.urlparse(redirect_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080

    httpd = http.server.HTTPServer((host, port), _CallbackHandler)
    httpd.query_params = {}  # type: ignore[attr-defined]
    eprint(f"Waiting for OAuth callback on {host}:{port} ...")
    httpd.handle_request()

    qs = httpd.query_params  # type: ignore[attr-defined]
    if "error" in qs:
        raise RuntimeError(f"OAuth error: {qs.get('error')} {qs.get('error_description')}")
    code = (qs.get("code") or [None])[0]
    got_state = (qs.get("state") or [None])[0]
    if not code:
        raise RuntimeError("No code returned in OAuth callback.")
    if got_state != state:
        raise RuntimeError("State mismatch in OAuth callback.")

    # Exchange code for token
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code": code,
        "code_verifier": verifier,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if api_key_header_name:
        headers[api_key_header_name] = client_id

    resp = requests.post(token_url, data=data, headers=headers, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Token exchange failed: {resp.status_code} {resp.text}")
    return resp.json()


# -----------------------------
# Etsy API client (framework)
# -----------------------------

class EtsyClient:
    def __init__(self, api_base: str, client_id: str, access_token: str):
        self.api_base = api_base.rstrip("/")
        self.client_id = client_id
        self.access_token = access_token
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        # Etsy commonly requires x-api-key and Authorization Bearer; confirm in docs and adjust.
        return {
            "Authorization": f"Bearer {self.access_token}",
            "x-api-key": self.client_id,
            "Accept": "application/json",
        }

    def request(self, method: str, path: str, *, json_body: Any = None, files: Any = None) -> Any:
        url = f"{self.api_base}{path}"
        for attempt in range(6):
            resp = self.session.request(
                method=method,
                url=url,
                headers=self._headers(),
                json=json_body if files is None else None,
                files=files,
                timeout=60,
            )

            # Handle rate limits / transient errors
            if resp.status_code in (429, 500, 502, 503, 504):
                backoff_sleep(attempt)
                continue

            if resp.status_code >= 300:
                raise RuntimeError(f"API error {resp.status_code} on {method} {path}: {resp.text}")

            if resp.text.strip():
                return resp.json()
            return None

        raise RuntimeError(f"API request failed after retries: {method} {path}")

    # ---- Listing operations (placeholders; adjust endpoints/payload to Etsy docs) ----

    def create_draft_listing(self, shop_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Example placeholder endpoint:
        # POST /shops/{shop_id}/listings
        return self.request("POST", f"/shops/{shop_id}/listings", json_body=payload)

    def update_listing(self, shop_id: int, listing_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("PUT", f"/shops/{shop_id}/listings/{listing_id}", json_body=payload)

    def upload_listing_image(self, shop_id: int, listing_id: int, image_path: Path, rank: int) -> Dict[str, Any]:
        # Many APIs accept multipart/form-data for images.
        # Adjust param names according to Etsy docs (e.g., "image", "rank", etc.)
        with image_path.open("rb") as f:
            files = {
                "image": (image_path.name, f, guess_mime(image_path)),
                "rank": (None, str(rank)),
            }
            return self.request("POST", f"/shops/{shop_id}/listings/{listing_id}/images", files=files)

    def update_inventory_variations(self, listing_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder. Etsy inventory endpoints often look like:
        # PUT /listings/{listing_id}/inventory
        return self.request("PUT", f"/listings/{listing_id}/inventory", json_body=payload)


# -----------------------------
# Payload builders (you must align to Etsy schema)
# -----------------------------

def build_listing_payload(p: Product) -> Dict[str, Any]:
    """
    Build a DRAFT listing payload.
    Adjust field names to match Etsy’s expected schema.
    """
    payload: Dict[str, Any] = {
        "title": p.title,
        "description": p.description,
        "price": f"{p.price:.2f}",
        "quantity": p.quantity,
        "who_made": p.who_made,
        "when_made": p.when_made,
        "is_supply": p.is_supply,
        # Draft-first: if Etsy supports a state field, set to "draft"
        "state": "draft",
    }
    if p.taxonomy_id is not None:
        payload["taxonomy_id"] = p.taxonomy_id
    if p.tags:
        payload["tags"] = p.tags
    if p.materials:
        payload["materials"] = p.materials
    if p.shipping_profile_id is not None:
        payload["shipping_profile_id"] = p.shipping_profile_id
    if p.return_policy_id is not None:
        payload["return_policy_id"] = p.return_policy_id
    return payload


def build_inventory_payload(p: Product) -> Optional[Dict[str, Any]]:
    """
    Build an inventory/variations payload.
    Etsy’s “inventory products/offerings/property_values” can be strict.
    You MUST align this builder to Etsy’s current inventory schema.
    """
    if not p.variants:
        return None

    # Generic shape (placeholder) - you must rewrite to Etsy docs.
    # The idea: property definitions (size/color/pattern) + offerings per variant
    properties = []
    for prop_name in ("size", "color", "pattern"):
        if p.variations and prop_name in p.variations:
            properties.append({
                "property_name": prop_name,
                "values": p.variations[prop_name],
            })

    offerings = []
    for v in p.variants:
        offerings.append({
            "sku": v.sku,
            "quantity": v.qty,
            "price": f"{(v.price or p.price):.2f}",
            "options": {
                "size": v.size,
                "color": v.color,
                "pattern": v.pattern,
            }
        })

    return {
        "properties": properties,
        "offerings": offerings,
    }


# -----------------------------
# Main flow
# -----------------------------

def run_upload(
    *,
    client: EtsyClient,
    shop_id: int,
    products: List[Product],
    images_dir: Path,
    dry_run: bool,
) -> None:
    for idx, p in enumerate(products, start=1):
        errs = validate_product(p)
        if errs:
            eprint(f"[{idx}/{len(products)}] SKIP (validation errors) title='{p.title}': {errs}")
            continue

        listing_payload = build_listing_payload(p)
        inventory_payload = build_inventory_payload(p)

        eprint(f"[{idx}/{len(products)}] Prepare: {p.title}")

        if dry_run:
            eprint("  DRY-RUN listing payload:", json.dumps(listing_payload, ensure_ascii=False)[:500], "...")
            if inventory_payload:
                eprint("  DRY-RUN inventory payload:", json.dumps(inventory_payload, ensure_ascii=False)[:500], "...")
            continue

        # 1) Create draft listing
        created = client.create_draft_listing(shop_id, listing_payload)

        # You must confirm the returned field name for listing id in Etsy response.
        listing_id = created.get("listing_id") or created.get("id")
        if not listing_id:
            raise RuntimeError(f"Could not find listing_id in response: {created}")
        listing_id = int(listing_id)
        eprint(f"  Created listing_id={listing_id}")

        # 2) Set inventory/variations (if any)
        if inventory_payload:
            try:
                inv_resp = client.update_inventory_variations(listing_id, inventory_payload)
                eprint("  Inventory updated.")
            except Exception as ex:
                eprint(f"  WARNING: inventory update failed: {ex}")

        # 3) Upload images in order
        for rank, img_name in enumerate(p.images, start=1):
            img_path = images_dir / img_name
            if not img_path.exists():
                eprint(f"  WARNING: image not found, skip: {img_path}")
                continue
            try:
                client.upload_listing_image(shop_id, listing_id, img_path, rank)
                eprint(f"  Uploaded image rank={rank}: {img_name}")
            except Exception as ex:
                eprint(f"  WARNING: image upload failed for {img_name}: {ex}")

        eprint(f"  Done: {p.title}")


def ensure_token(config: Dict[str, Any]) -> str:
    token_file = Path(config.get("token_file", "etsy_tokens.json"))
    if token_file.exists():
        tok = load_json(token_file)
        access = tok.get("access_token")
        if access:
            return access

    # Interactive OAuth (you MUST fill proper Etsy auth/token URLs)
    auth_url = config.get("auth_url", "").strip()
    token_url = config.get("token_url", "").strip()
    if not auth_url or not token_url:
        raise SystemExit(
            "Missing auth_url/token_url in config. Add Etsy OAuth authorize URL and token URL.\n"
            "Tip: Put them in config.json as 'auth_url' and 'token_url'."
        )

    token = oauth_authorize_interactive(
        client_id=config["client_id"],
        redirect_uri=config["redirect_uri"],
        scopes=config.get("scopes", []),
        auth_url=auth_url,
        token_url=token_url,
        api_key_header_name=config.get("api_key_header_name"),  # optional
    )
    save_json(token_file, token)
    eprint(f"Saved token to {token_file}")
    return token["access_token"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config.json path")
    ap.add_argument("--input", required=True, help="products.json or products.csv")
    ap.add_argument("--images", required=True, help="directory containing images")
    ap.add_argument("--dry-run", action="store_true", help="validate & print payloads only")
    args = ap.parse_args()

    config = load_json(args.config)
    images_dir = Path(args.images)
    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    if inp.suffix.lower() == ".json":
        products = parse_products_from_json(inp)
    elif inp.suffix.lower() == ".csv":
        products = parse_products_from_csv(inp)
    else:
        raise SystemExit("Input must be .json or .csv")

    access_token = ensure_token(config)

    client = EtsyClient(
        api_base=config.get("api_base", "https://api.etsy.com/v3/application"),
        client_id=config["client_id"],
        access_token=access_token,
    )

    shop_id = int(config["shop_id"])
    run_upload(client=client, shop_id=shop_id, products=products, images_dir=images_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
