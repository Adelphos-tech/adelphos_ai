import os
import re
import asyncio
import time as _time
from functools import lru_cache
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "NEW_PROPERTIES_S")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Initialize once at import time — async client for non-blocking search
_client = AsyncQdrantClient(url=QDRANT_URL)
_model = SentenceTransformer(EMBED_MODEL)

print(f"[QDRANT] Connected to {QDRANT_URL}, collection={COLLECTION_NAME}, model={EMBED_MODEL}")


@lru_cache(maxsize=256)
def _encode_cached(text: str) -> tuple:
    """CPU-bound embedding with LRU cache. Returns tuple (hashable) for caching."""
    return tuple(_model.encode(text).tolist())


# Warmup: pre-compute one embedding to JIT-compile model internals
_warmup_start = _time.time()
_encode_cached("warmup query for Singapore property search")
print(f"[QDRANT] Model warmed up in {int((_time.time()-_warmup_start)*1000)}ms")


async def _encode_async(text: str) -> list[float]:
    """Run SentenceTransformer encoding in thread executor to avoid blocking."""
    loop = asyncio.get_event_loop()
    vec = await loop.run_in_executor(None, _encode_cached, text)
    return list(vec)


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


def _build_qdrant_filter(filters: dict) -> Filter | None:
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


async def search_properties(query: str, limit: int = 5) -> list[dict]:
    """
    Async hybrid search: vector similarity + payload filters.
    Falls back to pure vector search if filtered search returns too few results.
    """
    t0 = _time.time()

    vector = await _encode_async(query)
    t_enc = _time.time()

    filters = _extract_filters(query)
    qdrant_filter = _build_qdrant_filter(filters)

    results = await _client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=qdrant_filter,
        limit=limit * 3,
        with_payload=True,
    )

    # Fallback to unfiltered if too few results
    if len(results) < 2 and qdrant_filter:
        print(f"[QDRANT] Filtered search returned {len(results)}, falling back to unfiltered")
        results = await _client.search(
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
            "image_url": p.get("image_url", ""),
            "all_image_urls": p.get("all_image_urls", []),
            "url": p.get("url", ""),
            "score": round(hit.score, 3),
        }
        properties.append(prop)

    t_done = _time.time()
    print(f"[QDRANT] Query: '{query[:60]}' → {len(properties)} results | encode={int((t_enc-t0)*1000)}ms search={int((t_done-t_enc)*1000)}ms total={int((t_done-t0)*1000)}ms")
    return properties


def format_properties_for_llm(properties: list[dict]) -> str:
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
