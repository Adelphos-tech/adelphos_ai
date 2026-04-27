import os
import re
import asyncio
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "")
_qdrant_available = bool(QDRANT_URL)

if _qdrant_available:
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[QDRANT] sentence-transformers not installed. Mock mode only.")
            raise Exception("sentence-transformers not available")
        import time as _time
        COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "NEW_PROPERTIES_S")
        EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        print(f"[QDRANT] Loading embedding model {EMBED_MODEL}...")
        _t0 = _time.time()
        _model: SentenceTransformer = SentenceTransformer(EMBED_MODEL)
        _model.encode("singapore property", convert_to_numpy=True)
        print(f"[QDRANT] Model ready in {int((_time.time()-_t0)*1000)}ms")
        _client = None
        def _get_client() -> AsyncQdrantClient:
            global _client
            if _client is None:
                _client = AsyncQdrantClient(url=QDRANT_URL)
            return _client
        @lru_cache(maxsize=256)
        def _encode_cached(text: str) -> tuple:
            return tuple(_model.encode(text).tolist())
        async def _encode_async(text: str):
            loop = asyncio.get_event_loop()
            return list(await loop.run_in_executor(None, _encode_cached, text))
    except Exception as e:
        print(f"[QDRANT] Failed to load: {e}. Property search disabled.")
        _qdrant_available = False
else:
    print("[QDRANT] No QDRANT_URL set — property search disabled. Using mock properties for demo.")

# ─── Mock Properties for Demo (when Qdrant is not configured) ───
MOCK_PROPERTIES = [
    {
        "title": "Luxury Condo at Marina One Residences",
        "address": "21 Marina Way",
        "district": "D01 - Marina Bay",
        "category": "for-sale",
        "property_type": "Condominium",
        "bedrooms": 3,
        "bathrooms": 2,
        "floor_area": 1200,
        "floor_area_raw": "1,200 sqft",
        "price": 3200000,
        "display_price": "SGD 3,200,000",
        "price_raw": "SGD3,200,000",
        "psf": "SGD 2,667 psf",
        "tenure": "99-year Leasehold",
        "listed_date": "Jan 2025",
        "agent_name": "Mohamed Habib",
        "agent_agency": "APIL Properties",
        "description": "Stunning 3-bedroom unit at Marina One Residences with breathtaking views of Marina Bay. Premium finishes, smart home features, and access to world-class facilities including sky gardens and swimming pools.",
        "image_url": "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?w=800&h=600&fit=crop",
        "all_image_urls": [
            "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=800&h=600&fit=crop"
        ],
        "url": "https://apilproperties.com/listings/marina-one",
        "score": 0.95,
    },
    {
        "title": "Modern HDB BTO at Tampines GreenVerge",
        "address": "Tampines Street 86",
        "district": "D18 - Tampines",
        "category": "for-sale",
        "property_type": "HDB",
        "bedrooms": 4,
        "bathrooms": 2,
        "floor_area": 1100,
        "floor_area_raw": "1,100 sqft",
        "price": 680000,
        "display_price": "SGD 680,000",
        "price_raw": "SGD680,000",
        "psf": "SGD 618 psf",
        "tenure": "99-year Leasehold",
        "listed_date": "Feb 2025",
        "agent_name": "Mohamed Habib",
        "agent_agency": "APIL Properties",
        "description": "Spacious 4-room HDB BTO flat in the vibrant Tampines estate. Close to Tampines Mall, Century Square, and MRT station. Modern design with efficient layout perfect for families.",
        "image_url": "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?w=800&h=600&fit=crop",
        "all_image_urls": [
            "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=800&h=600&fit=crop"
        ],
        "url": "https://apilproperties.com/listings/tampines-bto",
        "score": 0.88,
    },
    {
        "title": "Prestigious Landed Home at Bukit Timah",
        "address": "Jalan Serene",
        "district": "D11 - Bukit Timah",
        "category": "for-sale",
        "property_type": "Semi-Detached",
        "bedrooms": 5,
        "bathrooms": 4,
        "floor_area": 3500,
        "floor_area_raw": "3,500 sqft",
        "price": 6500000,
        "display_price": "SGD 6,500,000",
        "price_raw": "SGD6,500,000",
        "psf": "SGD 1,857 psf",
        "tenure": "Freehold",
        "listed_date": "Jan 2025",
        "agent_name": "Mohamed Habib",
        "agent_agency": "APIL Properties",
        "description": "Elegant semi-detached house in the prestigious Bukit Timah estate. Features a private pool, home lift, and lush garden. Near top schools like Nanyang Primary and Raffles Girls.",
        "image_url": "https://images.unsplash.com/photo-1613490493576-7fde63acd811?w=800&h=600&fit=crop",
        "all_image_urls": [
            "https://images.unsplash.com/photo-1613490493576-7fde63acd811?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1600585154526-990dced4db0d?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?w=800&h=600&fit=crop"
        ],
        "url": "https://apilproperties.com/listings/bukit-timah-landed",
        "score": 0.92,
    },
    {
        "title": "Cozy Studio at Orchard Residences",
        "address": "238 Orchard Boulevard",
        "district": "D09 - Orchard",
        "category": "for-rent",
        "property_type": "Condominium",
        "bedrooms": 1,
        "bathrooms": 1,
        "floor_area": 500,
        "floor_area_raw": "500 sqft",
        "price": 3800,
        "display_price": "SGD 3,800/mo",
        "price_raw": "SGD3,800/month",
        "psf": "SGD 7.60 psf",
        "tenure": "99-year Leasehold",
        "listed_date": "Mar 2025",
        "agent_name": "Mohamed Habib",
        "agent_agency": "APIL Properties",
        "description": "Modern studio apartment in the heart of Orchard Road. Direct access to ION Orchard and MRT. Fully furnished with high-end appliances. Perfect for professionals.",
        "image_url": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=800&h=600&fit=crop",
        "all_image_urls": [
            "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0?w=800&h=600&fit=crop"
        ],
        "url": "https://apilproperties.com/listings/orchard-studio",
        "score": 0.85,
    },
    {
        "title": "3-Bedroom HDB Flat for Rent at Punggol",
        "address": "Sumang Walk, Punggol",
        "district": "D19 - Punggol",
        "category": "for-rent",
        "property_type": "HDB",
        "bedrooms": 3,
        "bathrooms": 2,
        "floor_area": 1000,
        "floor_area_raw": "1,000 sqft",
        "price": 3200,
        "display_price": "SGD 3,200/mo",
        "price_raw": "SGD3,200/month",
        "psf": "SGD 3.20 psf",
        "tenure": "99-year Leasehold",
        "listed_date": "Feb 2025",
        "agent_name": "Mohamed Habib",
        "agent_agency": "APIL Properties",
        "description": "Spacious 3-bedroom HDB flat in Punggol's waterfront district. Recently renovated with modern fittings. Walking distance to Punggol MRT, Waterway Point mall, and Punggol Waterway Park. Perfect for families.",
        "image_url": "https://images.unsplash.com/photo-1600566753151-384129cf4e3e?w=800&h=600&fit=crop",
        "all_image_urls": [
            "https://images.unsplash.com/photo-1600566753151-384129cf4e3e?w=800&h=600&fit=crop",
            "https://images.unsplash.com/photo-1600573472550-8090b5e0745e?w=800&h=600&fit=crop"
        ],
        "url": "https://apilproperties.com/listings/punggol-hdb-rent",
        "score": 0.90,
    },
    {
        "title": "Spacious HDB at Ang Mo Kio",
        "address": "Ang Mo Kio Avenue 10",
        "district": "D20 - Ang Mo Kio",
        "category": "for-rent",
        "property_type": "HDB",
        "bedrooms": 3,
        "bathrooms": 2,
        "floor_area": 900,
        "floor_area_raw": "900 sqft",
        "price": 3200,
        "display_price": "SGD 3,200/mo",
        "price_raw": "SGD3,200/month",
        "psf": "SGD 3.56 psf",
        "tenure": "99-year Leasehold",
        "listed_date": "Mar 2025",
        "agent_name": "Mohamed Habib",
        "agent_agency": "APIL Properties",
        "description": "Well-maintained 3-room HDB flat in mature Ang Mo Kio estate. Close to AMK Hub, MRT, and top schools. Partially furnished with renovated kitchen.",
        "image_url": "https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=800&h=600&fit=crop",
        "all_image_urls": [
            "https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=800&h=600&fit=crop"
        ],
        "url": "https://apilproperties.com/listings/amk-hdb",
        "score": 0.82,
    },
]

# ─── Singapore district/area alias map ───
LOCATION_ALIASES = {
    "orchard": ["orchard", "D09", "district 9"],
    "d9": ["orchard", "D09", "district 9"],
    "district 9": ["orchard", "D09", "district 9"],
    "marina bay": ["marina bay", "D01", "district 1"],
    "marina": ["marina bay", "marina", "D01"],
    "cbd": ["CBD", "central business district", "D01", "D02", "shenton"],
    "shenton": ["shenton", "D01"],
    "raffles": ["raffles", "D01"],
    "tanjong pagar": ["tanjong pagar", "D02"],
    "sentosa": ["sentosa", "D04"],
    "harbourfront": ["harbourfront", "D04"],
    "buona vista": ["buona vista", "D05"],
    "holland": ["holland", "D10"],
    "bukit timah": ["bukit timah", "D10", "D11"],
    "novena": ["novena", "D11"],
    "newton": ["newton", "D11"],
    "bishan": ["bishan", "D20"],
    "ang mo kio": ["ang mo kio", "D20", "AMK"],
    "amk": ["ang mo kio", "AMK"],
    "tampines": ["tampines", "D18"],
    "pasir ris": ["pasir ris", "D18"],
    "bedok": ["bedok", "D16"],
    "changi": ["changi", "D17"],
    "punggol": ["punggol", "D19"],
    "sengkang": ["sengkang", "D19"],
    "hougang": ["hougang", "D19"],
    "woodlands": ["woodlands", "D25"],
    "yishun": ["yishun", "D27"],
    "jurong": ["jurong", "D22"],
    "clementi": ["clementi", "D05"],
    "queenstown": ["queenstown", "D03"],
    "kallang": ["kallang", "D12"],
    "geylang": ["geylang", "D14"],
    "katong": ["katong", "D15"],
    "east coast": ["east coast", "D15", "D16"],
    "serangoon": ["serangoon", "D19"],
    "thomson": ["thomson", "D20"],
    "toa payoh": ["toa payoh", "D12"],
    "river valley": ["river valley", "D09"],
    "robertson quay": ["robertson", "D09"],
    "tiong bahru": ["tiong bahru", "D03"],
    "chinatown": ["chinatown", "D02"],
    "little india": ["little india", "D08"],
    "farrer": ["farrer", "D10"],
    "d1": ["district 1", "D01"],
    "d2": ["district 2", "D02"],
    "d3": ["district 3", "D03"],
    "d4": ["district 4", "D04"],
    "d5": ["district 5", "D05"],
    "d10": ["district 10", "D10"],
    "d11": ["district 11", "D11"],
    "d15": ["district 15", "D15"],
    "d16": ["district 16", "D16"],
    "d19": ["district 19", "D19"],
    "d25": ["district 25", "D25"],
}

PROPERTY_TYPES = {
    "hdb": "hdb",
    "condo": "condo",
    "condominium": "condo",
    "landed": "landed",
    "terrace": "terrace",
    "semi-detached": "semi-detached",
    "semi detached": "semi-detached",
    "bungalow": "bungalow",
    "apartment": "apartment",
    "penthouse": "penthouse",
    "studio": "studio",
    "executive condo": "executive condo",
    "ec": "executive condo",
}

LISTING_CATEGORIES = {
    "for sale": "for-sale",
    "sale": "for-sale",
    "buy": "for-sale",
    "for rent": "for-rent",
    "rent": "for-rent",
    "rental": "for-rent",
}


def _extract_filters(query: str) -> dict:
    """Extract structured filters from a natural language query."""
    q = query.lower()
    filters = {}

    # Extract bedrooms
    bed_match = re.search(r'(\d)\s*(?:bed(?:room)?s?|br|bhk|rm)', q)
    if bed_match:
        filters["bedrooms"] = int(bed_match.group(1))
    elif "studio" in q:
        filters["bedrooms"] = 0

    # Extract listing category (sale vs rent)
    for keyword, slug in LISTING_CATEGORIES.items():
        if keyword in q:
            filters["category_slug"] = slug
            break

    # Extract location
    for alias, terms in LOCATION_ALIASES.items():
        if alias in q:
            filters["location_terms"] = terms
            break

    # Extract max price (SGD)
    price_match = re.search(r'(?:under|below|max|budget|less than|within)\s*(?:sgd\s*)?\$?\s*([\d,.]+)\s*(million|m|k)?', q)
    if price_match:
        price_val = float(price_match.group(1).replace(',', ''))
        unit = (price_match.group(2) or "").lower()
        if unit in ("million", "m"):
            price_val *= 1_000_000
        elif unit == "k":
            price_val *= 1_000
        elif price_val < 100:
            price_val *= 1_000_000
        filters["max_price"] = price_val

    print(f"[QDRANT] Extracted filters: {filters}")
    return filters


def _build_qdrant_filter(filters: dict):
    """Build Qdrant Filter from extracted structured filters."""
    conditions = []

    if "bedrooms" in filters:
        conditions.append(
            FieldCondition(key="bedrooms", match=MatchValue(value=filters["bedrooms"]))
        )

    if "category_slug" in filters:
        conditions.append(
            FieldCondition(key="category_slug", match=MatchValue(value=filters["category_slug"]))
        )

    if "max_price" in filters:
        conditions.append(
            FieldCondition(key="price", range=Range(lte=filters["max_price"], gt=0))
        )

    if not conditions:
        return None
    return Filter(must=conditions)


def _filter_mock_properties(query: str, limit: int = 5):
    """Filter mock properties based on query keywords."""
    q = query.lower()
    filters = _extract_filters(query)

    results = []
    for p in MOCK_PROPERTIES:
        score = 0.5  # Base score

        # Check location match
        location_terms = filters.get("location_terms", [])
        if location_terms:
            loc_text = f"{p.get('district', '')} {p.get('address', '')}".lower()
            for term in location_terms:
                if term.lower() in loc_text:
                    score += 0.3
                    break

        # Check bedroom match
        if "bedrooms" in filters:
            if p.get("bedrooms") == filters["bedrooms"]:
                score += 0.2

        # Check category (sale/rent)
        if "category_slug" in filters:
            if p.get("category") == filters["category_slug"]:
                score += 0.2

        # Check price range
        if "max_price" in filters:
            price = p.get("price", 0) or 0
            if price <= filters["max_price"]:
                score += 0.1
            else:
                score -= 0.3  # Penalty for over budget

        # Keyword boosts
        keywords = ["condo", "hdb", "landed", "rent", "sale", "buy", "studio"]
        for kw in keywords:
            if kw in q and kw in p.get("title", "").lower():
                score += 0.1

        if score > 0.4:
            p_copy = dict(p)
            p_copy["score"] = round(min(score, 0.99), 3)
            results.append(p_copy)

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


async def search_properties(query: str, limit: int = 5):
    """
    Async hybrid search: vector similarity + payload filters.
    Falls back to pure vector search if filtered search returns too few results.
    Falls back to mock properties when Qdrant is not configured.
    """
    # Return mock properties if Qdrant is not available
    if not _qdrant_available:
        print(f"[QDRANT] Using mock properties for query: '{query[:50]}...'")
        return _filter_mock_properties(query, limit)

    t0 = _time.time()

    vector = await _encode_async(query)
    t_enc = _time.time()

    filters = _extract_filters(query)
    qdrant_filter = _build_qdrant_filter(filters)

    client = _get_client()
    results = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=qdrant_filter,
        limit=limit * 3,
        with_payload=True,
    )

    # Fallback to unfiltered if too few results
    if len(results) < 2 and qdrant_filter:
        print(f"[QDRANT] Filtered search returned {len(results)}, falling back to unfiltered")
        results = await client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=limit * 3,
            with_payload=True,
        )

    # Re-rank by location match if user specified a location
    location_terms = filters.get("location_terms", [])
    if location_terms:
        def location_boost(hit):
            p = hit.payload
            text = f"{p.get('district', '')} {p.get('address', '')} {p.get('title', '')} {p.get('content', '')}".lower()
            for term in location_terms:
                if term.lower() in text:
                    return hit.score + 0.5
            return hit.score
        results.sort(key=location_boost, reverse=True)

    properties = []
    seen_urls = set()
    for hit in results[:limit * 2]:
        p = hit.payload
        url = p.get("url", "") or p.get("url_slug", "")
        # Deduplicate by URL
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        if len(properties) >= limit:
            break

        price_raw = p.get("price") or 0
        try:
            price_display = f"SGD {int(float(price_raw)):,}" if price_raw else "Price on request"
        except (ValueError, TypeError):
            price_display = p.get("price_raw", "Price on request")

        # Use price_raw string if available for display (e.g. "SGD2218888")
        if p.get("price_raw") and price_raw:
            price_display = p.get("price_raw", price_display)

        prop = {
            "title": p.get("title", ""),
            "address": p.get("address", ""),
            "district": p.get("district", ""),
            "category": p.get("category", ""),
            "property_type": p.get("property_type", ""),
            "bedrooms": p.get("bedrooms", ""),
            "bathrooms": p.get("bathrooms", ""),
            "floor_area": p.get("floor_area", ""),
            "floor_area_raw": p.get("floor_area_raw", ""),
            "price": price_raw,
            "display_price": price_display,
            "price_raw": p.get("price_raw", ""),
            "psf": p.get("raw_details", {}).get("Psf", "") if isinstance(p.get("raw_details"), dict) else "",
            "tenure": p.get("tenure", ""),
            "listed_date": p.get("listed_date", ""),
            "agent_name": p.get("agent_name", ""),
            "agent_agency": p.get("agent_agency", ""),
            "description": (p.get("description", "") or "")[:300],
            "image_url": next((u for u in (p.get("all_image_urls") or []) if "/agent/" not in u), ""),
            "all_image_urls": [u for u in (p.get("all_image_urls") or []) if "/agent/" not in u],
            "agent_image": next((u for u in (p.get("all_image_urls") or []) if "/agent/" in u), p.get("image_url", "")),
            "url": p.get("url", ""),
            "score": round(hit.score, 3),
        }
        properties.append(prop)

    t_done = _time.time()
    print(f"[QDRANT] Query: '{query[:60]}' → {len(properties)} results | encode={int((t_enc-t0)*1000)}ms search={int((t_done-t_enc)*1000)}ms total={int((t_done-t0)*1000)}ms")
    return properties


async def ingest_properties_from_file(file_path: str) -> dict:
    """
    Ingest properties from an Excel or CSV file into Qdrant.
    Expected columns: title, address, district, price, bedrooms, bathrooms,
    floor_area, property_type, category, tenure, description, url, image_url, agent_name, agent_agency
    """
    import pandas as pd
    import uuid as _uuid
    from qdrant_client.models import PointStruct

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    required = ['title']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Missing required columns: {missing}", "total_rows": len(df), "inserted": 0}

    model = await _get_model()
    client = _get_client()

    points = []
    errors = 0
    for idx, row in df.iterrows():
        try:
            title = str(row.get('title', '')).strip()
            if not title:
                errors += 1
                continue

            # Build text for embedding
            parts = [title]
            for col in ['address', 'district', 'property_type', 'category', 'description']:
                val = str(row.get(col, '')).strip()
                if val and val.lower() != 'nan':
                    parts.append(val)
            embed_text = ' '.join(parts)

            vector = model.encode(embed_text).tolist()

            payload = {}
            for col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    payload[col] = str(val).strip() if isinstance(val, str) else val

            # Handle image URLs
            if 'image_url' in payload and isinstance(payload['image_url'], str):
                payload['all_image_urls'] = [u.strip() for u in payload['image_url'].split(',') if u.strip()]

            point_id = str(_uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
        except Exception as e:
            print(f"[QDRANT INGEST] Row {idx} error: {e}")
            errors += 1

    if points:
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            await client.upsert(collection_name=COLLECTION_NAME, points=batch)
            print(f"[QDRANT INGEST] Inserted batch {i//batch_size + 1} ({len(batch)} points)")

    return {
        "message": "Property data ingested successfully",
        "total_rows": len(df),
        "inserted": len(points),
        "errors": errors,
    }


def format_properties_for_llm(properties) -> str:
    """Format Singapore property search results as context for the LLM."""
    if not properties:
        return ""

    lines = []
    for i, p in enumerate(properties, 1):
        price_str = p.get("price_raw") or p.get("display_price", "Price on request")
        bedrooms = p.get("bedrooms", "N/A")
        bathrooms = p.get("bathrooms", "N/A")
        area = p.get("floor_area_raw") or (f"{p.get('floor_area')} sqft" if p.get("floor_area") else "")
        district = p.get("district", "")
        address = p.get("address", "")
        location = f"{address}, {district}".strip(", ") or "Singapore"
        ptype = p.get("property_type", "") or p.get("category", "")
        tenure = p.get("tenure", "")
        agent = p.get("agent_name", "")
        agency = p.get("agent_agency", "")
        psf = p.get("psf", "")
        url = p.get("url", "")
        description = p.get("description", "")

        parts = [
            f"{i}. {p['title']}",
            f"Category: {p.get('category', '')}",
            f"Type: {ptype}",
            f"Bedrooms: {bedrooms}",
            f"Bathrooms: {bathrooms}",
            f"Price: {price_str}",
            f"Location: {location}",
        ]
        if area:
            parts.append(f"Size: {area}")
        if psf:
            parts.append(f"PSF: {psf}")
        if tenure:
            parts.append(f"Tenure: {tenure}")
        if agent:
            parts.append(f"Agent: {agent}{' (' + agency + ')' if agency else ''}")
        if description:
            parts.append(f"About: {description[:150]}")
        if url:
            parts.append(f"Link: {url}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)
