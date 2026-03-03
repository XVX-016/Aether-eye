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
