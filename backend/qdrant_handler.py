import os
import re
import asyncio
import time as _time
from functools import lru_cache
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, MatchText
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    print("[QDRANT] httpx not available — image fetching disabled")

APIL_IMAGE_BASE = "https://d1up4ebiscsd6l.cloudfront.net/"
APIL_PROPERTY_API = "https://admin.apilproperties.com/api/property/"

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "NEW_PROPERTIES_S")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Initialize once at import time — async client for non-blocking search
_client = AsyncQdrantClient(url=QDRANT_URL)
_model = SentenceTransformer(EMBED_MODEL)
_executor_loop = None  # set lazily

print(f"[QDRANT] Connected to {QDRANT_URL}, collection={COLLECTION_NAME}, model={EMBED_MODEL}")


@lru_cache(maxsize=256)
def _encode_cached(text: str) -> tuple:
    """CPU-bound embedding with LRU cache. Returns tuple (hashable) for caching."""
    return tuple(_model.encode(text).tolist())


# Warmup: pre-compute one embedding to JIT-compile model internals (~300ms saved on first real query)
_warmup_start = _time.time()
_encode_cached("warmup query for property search")
print(f"[QDRANT] Model warmed up in {int((_time.time()-_warmup_start)*1000)}ms")


async def _encode_async(text: str) -> list[float]:
    """Run SentenceTransformer encoding in thread executor to avoid blocking."""
    loop = asyncio.get_event_loop()
    vec = await loop.run_in_executor(None, _encode_cached, text)
    return list(vec)

# ─── Location alias map: common names → Qdrant field substrings ───
LOCATION_ALIASES = {
    "dubai marina": ["Marsa Dubai", "Dubai Marina"],
    "marina": ["Marsa Dubai", "Dubai Marina"],
    "jbr": ["Jumeirah Beach Resid", "JBR"],
    "jumeirah beach": ["Jumeirah Beach"],
    "downtown": ["Burj Khalifa", "Downtown Dubai"],
    "business bay": ["Business Bay", "Al Khaleej"],
    "jvc": ["Al Barsha South Fourth", "Jumeirah village circle", "JVC"],
    "jumeirah village circle": ["Al Barsha South Fourth", "Jumeirah village circle"],
    "jvt": ["Jumeirah Village Triangle"],
    "palm": ["Palm Jumeirah"],
    "palm jumeirah": ["Palm Jumeirah"],
    "creek harbour": ["Dubai Creek Harbour", "Al khairan"],
    "dubai creek": ["Dubai Creek Harbour", "Al khairan"],
    "dubai hills": ["Dubai Hills Estate", "Hadaeq Sheikh"],
    "arabian ranches": ["Arabian Ranches"],
    "motor city": ["Dubai Motor City"],
    "dubai south": ["Dubai South"],
    "al furjan": ["Al Furjan"],
    "arjan": ["Arjan", "Al Barshaa South"],
    "dubai islands": ["Dubai Islands"],
    "sobha hartland": ["Sobha Hartland"],
    "al barari": ["Al Barari"],
    "damac hills": ["DAMAC"],
    "ras al khaimah": ["Ras Al Khaimah", "Al Hamra"],
    "marjan island": ["Al Marjan Island"],
    "science park": ["Dubai Science Park"],
    "silicon oasis": ["Silicon"],
    "sports city": ["Sports City"],
    "meadows": ["Meadows"],
    "springs": ["Springs"],
    "al jaddaf": ["Al Jaddaf", "Al Jadaf"],
}

PROPERTY_TYPES = {
    "villa": "Villa",
    "apartment": "Apartment",
    "townhouse": "Townhouse",
    "penthouse": "Penthouse",
    "studio": "Apartment",
    "flat": "Apartment",
    "duplex": "Duplex",
}


def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', ' ', text).strip()


def _extract_filters(query: str) -> dict:
    """Extract structured filters from a natural language query."""
    q = query.lower()
    filters = {}

    # Extract bedrooms
    bed_match = re.search(r'(\d)\s*(?:bed(?:room)?s?|br|bhk)', q)
    if bed_match:
        filters["bedrooms"] = int(bed_match.group(1))
    elif "studio" in q:
        filters["bedrooms"] = 0

    # Extract property type
    for keyword, ptype in PROPERTY_TYPES.items():
        if keyword in q:
            filters["property_type"] = ptype
            break

    # Extract location
    for alias, qdrant_values in LOCATION_ALIASES.items():
        if alias in q:
            filters["location_terms"] = qdrant_values
            break

    # Extract max price
    price_match = re.search(r'(?:under|below|max|budget|less than)\s*(?:aed\s*)?([\d,.]+)\s*(million|m|k)?', q)
    if price_match:
        price_val = float(price_match.group(1).replace(',', ''))
        unit = (price_match.group(2) or "").lower()
        if unit in ("million", "m"):
            price_val *= 1_000_000
        elif unit == "k":
            price_val *= 1_000
        elif price_val < 100:  # likely millions
            price_val *= 1_000_000
        filters["max_price"] = price_val

    print(f"[QDRANT] Extracted filters: {filters}")
    return filters


def _build_qdrant_filter(filters: dict) -> Filter | None:
    """Build Qdrant Filter from extracted structured filters."""
    conditions = []

    if "bedrooms" in filters:
        conditions.append(
            FieldCondition(key="bedroom_norm", match=MatchValue(value=filters["bedrooms"]))
        )

    if "property_type" in filters:
        conditions.append(
            FieldCondition(key="property_type", match=MatchValue(value=filters["property_type"]))
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

    # Run encoding in thread executor (CPU-bound, ~30ms cached / ~300ms uncached)
    vector = await _encode_async(query)
    t_enc = _time.time()

    filters = _extract_filters(query)
    qdrant_filter = _build_qdrant_filter(filters)

    # Async Qdrant search — non-blocking
    results = await _client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=qdrant_filter,
        limit=limit * 3,
        with_payload=True,
    )

    # If filtered search returned too few, fallback to unfiltered
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
            text = f"{p.get('district', '')} {p.get('community_area', '')} {p.get('address', '')} {p.get('name', '')}".lower()
            for term in location_terms:
                if term.lower() in text:
                    return hit.score + 0.5  # significant boost
            return hit.score
        results.sort(key=location_boost, reverse=True)

    properties = []
    seen_slugs = set()
    for hit in results[:limit * 2]:  # over-fetch to allow dedup
        p = hit.payload
        slug = p.get("slug", "")
        # Deduplicate by slug (each property may have multiple chunks)
        if slug and slug in seen_slugs:
            continue
        if slug:
            seen_slugs.add(slug)
        if len(properties) >= limit:
            break

        name = p.get("project_name") or p.get("name", "")
        community = p.get("community_area", "")
        district = p.get("district", "")
        full_location = community or district or _strip_html(p.get("address", ""))

        price_raw = p.get("price") or p.get("sale_price") or 0
        try:
            price_display = f"AED {int(float(price_raw)):,}" if price_raw else "Price on request"
        except (ValueError, TypeError):
            price_display = str(price_raw)

        prop = {
            "name": name,
            "slug": slug,
            "property_id": p.get("property_id", p.get("id", "")),
            "category": p.get("category", p.get("property_type", "")),
            "property_type": p.get("property_type", p.get("category", "")),
            "bedroom": p.get("bedroom") or p.get("bedrooms", ""),
            "bedrooms": p.get("bedrooms") or p.get("bedroom", ""),
            "bathroom": p.get("bathroom") or p.get("bathrooms", ""),
            "bathrooms": p.get("bathrooms") or p.get("bathroom", ""),
            "price": price_raw,
            "display_price": price_display,
            "size_sq_ft": p.get("size_sq_ft") or p.get("area_sqft", ""),
            "parking": p.get("parking", ""),
            "capital_roi": p.get("capital_roi", ""),
            "listing_type": p.get("listing_type", ""),
            "full_location": full_location,
            "community_area": community,
            "district": district,
            "city": p.get("city", ""),
            "developer": p.get("developer", ""),
            "status": p.get("status") or p.get("completion_status", ""),
            "address": _strip_html(p.get("address", "")),
            "detail_url": f"https://apilproperties.com/property-detail/{slug}" if slug else "",
            "score": round(hit.score, 3),
        }
        properties.append(prop)

    # Fetch images in parallel from admin API
    if _HTTPX_AVAILABLE and properties:
        await _attach_images(properties)

    t_done = _time.time()
    print(f"[QDRANT] Query: '{query[:60]}' → {len(properties)} results | encode={int((t_enc-t0)*1000)}ms search={int((t_done-t_enc)*1000)}ms total={int((t_done-t0)*1000)}ms")
    return properties


async def _fetch_image_url(client: "httpx.AsyncClient", slug: str) -> str:
    """Fetch first image URL for a property slug from admin API. Returns full URL or empty string."""
    if not slug:
        return ""
    try:
        resp = await client.get(f"{APIL_PROPERTY_API}{slug}", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            images = data.get("images", [])
            if images:
                img = images[0]
                url = img.get("url", "") if isinstance(img, dict) else str(img)
                if url and not url.startswith("http"):
                    url = APIL_IMAGE_BASE + url
                return url
    except Exception:
        pass
    return ""


async def _attach_images(properties: list[dict]) -> None:
    """Fetch images for all properties in parallel and attach image_url field."""
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_image_url(client, p.get("slug", "")) for p in properties]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for prop, result in zip(properties, results):
        prop["image_url"] = result if isinstance(result, str) else ""
        if prop["image_url"]:
            print(f"[QDRANT] Image fetched for '{prop['name'][:40]}'")
        else:
            print(f"[QDRANT] No image for '{prop['name'][:40]}'")


def format_properties_for_llm(properties: list[dict]) -> str:
    """Format property search results as rich context for the LLM."""
    if not properties:
        return ""

    lines = []
    for i, p in enumerate(properties, 1):
        price_str = p.get("display_price", "Price on request")
        bedrooms = p.get("bedrooms") or p.get("bedroom", "N/A")
        bathrooms = p.get("bathrooms") or p.get("bathroom", "N/A")
        location = p.get("full_location") or p.get("community_area") or p.get("district", "Dubai")
        developer = p.get("developer", "")
        status = p.get("status", "")
        size = p.get("size_sq_ft", "")
        ptype = p.get("category") or p.get("property_type", "")
        detail_url = p.get("detail_url", "")

        parts = [
            f"{i}. {p['name']}",
            f"Type: {ptype}",
            f"Bedrooms: {bedrooms}",
            f"Bathrooms: {bathrooms}",
            f"Price: {price_str}",
            f"Location: {location}",
        ]
        if developer:
            parts.append(f"Developer: {developer}")
        if status:
            parts.append(f"Status: {status}")
        if size:
            parts.append(f"Size: {size} sqft")
        if detail_url:
            parts.append(f"Link: {detail_url}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)
