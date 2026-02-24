# estytool
Private internal app for my own Etsy shop selling handmade knitted physical goods. Uses Etsy API to bulk create/update listings (draft-first), manage variations (size/color/pattern with SKUs, qty, pricing), and batch upload/order photos

# Etsy Bulk Listing Uploader (Private Internal Tool)

This project contains **one Python file**: `etsy_bulk_uploader.py`.

It is a **private/internal tool** designed for **one Etsy shop owner** to:
- **Bulk create listings** (draft-first workflow)
- **Batch upload and order listing images**
- **Manage variations** for physical handmade products (e.g., **size / color / pattern**) with **SKU + quantity + optional price per variant**
- **Batch update inventory/restock data** (where supported by the Etsy API)

> Note: Etsy’s exact API endpoints and the inventory/variation payload schema can differ by API version.  
> The script is a solid framework and includes clearly marked places where we must align with Etsy’s current documentation for:
> - OAuth authorization URLs (`auth_url`, `token_url`)
> - Listing endpoints (create/update)
> - Inventory/variations payload shape

---

## Requirements

- Python **3.9+** recommended
- Internet access
- An **Etsy Developer App** (Client ID / API keystring)
- Approved OAuth scopes for listing write access

Python dependency:
- `requests`

Install dependency:
```bash
pip install requests

## Run Commands (Upload Listings + Images)

### A) Recommended: Validate first (no API calls)
This checks your input and shows payload previews without uploading anything:

```bash
python etsy_bulk_uploader.py --config config.json --input products.json --images ./images --dry-run
