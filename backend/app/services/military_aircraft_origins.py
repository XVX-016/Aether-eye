MILITARY_AIRCRAFT_ORIGINS = {
    # USA
    "F16": "USA", "F15": "USA", "F18": "USA", "F22": "USA",
    "F35": "USA", "F117": "USA", "F14": "USA", "F4": "USA",
    "B1": "USA", "B2": "USA", "B21": "USA", "B52": "USA",
    "A10": "USA", "C130": "USA", "C17": "USA", "C5": "USA",
    "KC135": "USA", "E2": "USA", "E7": "USA", "P3": "USA",
    "U2": "USA", "SR71": "USA", "MQ9": "USA", "RQ4": "USA",
    "MQ25": "USA", "UH60": "USA", "AH64": "USA", "CH47": "USA",
    "CH53": "USA", "V22": "USA", "AV8B": "USA", "C2": "USA",
    "YF23": "USA", "XB70": "USA", "X29": "USA", "X32": "USA",
    "XQ58": "USA",
    # Russia
    "Su24": "Russia", "Su25": "Russia", "Su34": "Russia",
    "Su57": "Russia", "Su47": "Russia", "Su27": "Russia",
    "Mig29": "Russia", "Mig31": "Russia", "Tu22M": "Russia",
    "Tu95": "Russia", "Tu160": "Russia", "Il76": "Russia",
    "Ka52": "Russia", "Ka27": "Russia", "Mi24": "Russia",
    "Mi28": "Russia", "Mi8": "Russia", "Mi26": "Russia",
    "An124": "Russia", "An22": "Russia", "An72": "Russia",
    "An225": "Russia",
    # China
    "J10": "China", "J20": "China", "J35": "China", "J36": "China",
    "J50": "China", "JH7": "China", "H6": "China", "Y20": "China",
    "WZ7": "China", "WZ10": "China", "WZ9": "China", "Z10": "China",
    "Z19": "China", "AG600": "China", "KJ600": "China", "KJ500": "China",
    # UK/Europe
    "Tornado": "UK", "Vulcan": "UK", "Hawk T1": "UK",
    "EF2000": "Germany",  # Eurofighter - multi-nation, mapped to Germany
    "Rafale": "France", "Mirage2000": "France",
    "JAS39": "Sweden", "SAAB340": "Sweden",
    "A400M": "Germany",  # Airbus multi-nation
    # Turkey
    "TB2": "Turkey", "AKINCI": "Turkey", "TB001": "Turkey",
    "KIZILELMA": "Turkey", "KAAN": "Turkey",
    # Japan
    "F2": "Japan", "T50": "Japan", "C1": "Japan", "US2": "Japan",
    "KF21": "Japan",  # actually South Korea
    # South Korea
    "FCK1": "South Korea", "KF21": "South Korea",
    # India
    "Tejas": "India",
    # Pakistan
    "JF17": "Pakistan",
    # Other
    "EMB314": "Brazil", "C390": "Brazil",
    "CL415": "Canada",
    "Be200": "Russia",
    "V280": "USA",
    "P8": "USA",
}

def get_aircraft_origin(folder_name: str) -> str:
    return MILITARY_AIRCRAFT_ORIGINS.get(folder_name, "Unknown")
