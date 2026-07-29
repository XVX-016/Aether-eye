from __future__ import annotations

from typing import Dict, List

COUNTRY_RELATIONS: Dict[str, Dict[str, List[str]]] = {
    "USA": {
        "allies": ["UK", "France", "Germany", "Japan"],
        "hostile": ["China", "Russia"],
    },
    "China": {
        "allies": ["Russia"],
        "hostile": ["USA", "Japan"],
    },
    "Russia": {
        "allies": ["China"],
        "hostile": ["USA", "UK", "France", "Germany"],
    },
    "India": {
        "allies": ["France"],
        "hostile": ["China"],
    },
    "UK": {
        "allies": ["USA", "France", "Germany"],
        "hostile": ["Russia"],
    },
    "France": {
        "allies": ["USA", "UK", "Germany", "India"],
        "hostile": ["Russia"],
    },
    "Germany": {
        "allies": ["USA", "UK", "France"],
        "hostile": ["Russia"],
    },
    "Japan": {
        "allies": ["USA"],
        "hostile": ["China", "Russia"],
    },
}


def classify_friend_foe(user_country: str, aircraft_origin_country: str) -> str:
    """
    Classify relationship between user country and aircraft origin country.
    Returns one of: FRIEND, FOE, NEUTRAL.
    """
    user = (user_country or "").strip()
    origin = (aircraft_origin_country or "").strip()

    if not user or not origin:
        return "NEUTRAL"
    if user == origin:
        return "FRIEND"

    relations = COUNTRY_RELATIONS.get(user, {})
    allies = set(relations.get("allies", []))
    hostile = set(relations.get("hostile", []))

    if origin in allies:
        return "FRIEND"
    if origin in hostile:
        return "FOE"
    return "NEUTRAL"


def get_origin_for_class(class_name: str) -> str:
    """
    Resolve origin country from either:
    - FGVC display name (F-16A/B, F/A-18, Eurofighter Typhoon)
    - Military dataset folder name (F16, F18, EF2000)
    """
    try:
        from app.services.military_aircraft_origins import get_aircraft_origin
    except ImportError:
        from services.military_aircraft_origins import get_aircraft_origin

    # Try direct folder name lookup first
    origin = get_aircraft_origin(class_name)
    if origin != "Unknown":
        return origin

    # FGVC name patterns - map to countries
    FGVC_ORIGINS = {
        "F-16A/B": "USA", "F/A-18": "USA", "F-15": "USA",
        "F-22": "USA", "F-35": "USA", "C-130": "USA",
        "Eurofighter Typhoon": "Germany", "Tornado": "UK",
        "Rafale": "France", "Mirage 2000": "France",
        "Il-76": "Russia", "An-12": "Russia",
        "Hawk T1": "UK",
    }
    return FGVC_ORIGINS.get(class_name, "Unknown")
