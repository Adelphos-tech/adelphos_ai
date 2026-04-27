import os
import re
from dotenv import load_dotenv

load_dotenv()

# For Render deployment - use mock data only (no heavy ML libraries)
_qdrant_available = False
print("[QDRANT] Using mock properties for demo mode.")

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
    Search properties - uses mock data for Render deployment.
    """
    # Always use mock properties for Render
    print(f"[QDRANT] Using mock properties for query: '{query[:50]}...'")
    return _filter_mock_properties(query, limit)


def format_properties_for_llm(properties):
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
