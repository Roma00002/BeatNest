import os
from typing import Dict, List, Optional

GENRE_STRUCTURE = {
    "hiphop_rap": {
        "name": "Hip-Hop / Rap",
        "subgenres": {
            "boom_bap": "Boom Bap",
            "trap": {
                "name": "Trap",
                "subgenres": {
                    "hood_trap": "Hood Trap",
                    "trap_soul": "Trap Soul",
                    "rage_trap": "Rage Trap",
                    "latin_trap": "Latin Trap"
                }
            },
            "drill": {
                "name": "Drill",
                "subgenres": {
                    "uk_drill": "UK Drill",
                    "ny_drill": "NY Drill",
                    "chicago_drill": "Chicago Drill"
                }
            },
            "lofi": "Lo-fi Hip-Hop",
            "cloud_rap": "Cloud Rap",
            "phonk": {
                "name": "Phonk",
                "subgenres": {
                    "drift_phonk": "Drift Phonk",
                    "memphis_phonk": "Memphis Phonk"
                }
            }
        }
    },
    "electronica": {
        "name": "Electrónica",
        "subgenres": {
            "edm": {
                "name": "EDM",
                "subgenres": {
                    "big_room": "Big Room",
                    "electro_house": "Electro House",
                    "future_house": "Future House"
                }
            },
            "trap_edm": "Trap EDM / Festival Trap",
            "dubstep": {
                "name": "Dubstep",
                "subgenres": {
                    "riddim": "Riddim",
                    "brostep": "Brostep"
                }
            },
            "chillstep": "Chillstep",
            "drum_bass": {
                "name": "Drum and Bass",
                "subgenres": {
                    "liquid_dnb": "Liquid DnB",
                    "neurofunk": "Neurofunk"
                }
            },
            "house": {
                "name": "House",
                "subgenres": {
                    "deep_house": "Deep House",
                    "tech_house": "Tech House",
                    "progressive_house": "Progressive House"
                }
            },
            "synthwave": "Synthwave",
            "future_bass": "Future Bass",
            "jersey_club": "Jersey Club"
        }
    },
    "latino": {
        "name": "Latino / Mundial",
        "subgenres": {
            "reggaeton": {
                "name": "Reggaetón",
                "subgenres": {
                    "reggaeton_viejo": "Reggaetón Viejo",
                    "reggaeton_romantico": "Reggaetón Romántico",
                    "trapeton": "Trapeton"
                }
            },
            "dembow": "Dembow",
            "dancehall": "Dancehall",
            "afrobeat": {
                "name": "Afrobeat",
                "subgenres": {
                    "afroswing": "Afroswing",
                    "afrotrap": "Afrotrap"
                }
            },
            "cumbia": "Cumbia Electrónica",
            "moombahton": "Moombahton"
        }
    },
    "otros": {
        "name": "Otros Géneros",
        "subgenres": {
            "rnb": {
                "name": "R&B",
                "subgenres": {
                    "alt_rnb": "Alternative R&B",
                    "neo_soul": "Neo Soul"
                }
            },
            "pop_urbano": "Pop Urbano",
            "experimental": "Experimental",
            "industrial_hiphop": "Industrial Hip-Hop",
            "glitch_hop": "Glitch Hop"
        }
    }
}

def get_genre_path(genre_path: str) -> Optional[str]:
    """
    Get the full path for a genre/subgenre.
    
    Args:
        genre_path (str): Path in format "genre/subgenre" or "genre/subgenre/subsubgenre"
        
    Returns:
        Optional[str]: Full path to genre directory if valid, None otherwise
    """
    parts = genre_path.split('/')
    current = GENRE_STRUCTURE
    
    # Navigate through the genre structure
    for part in parts:
        part = part.lower().replace(' ', '_')
        if part in current:
            current = current[part]
        elif 'subgenres' in current and part in current['subgenres']:
            current = current['subgenres'][part]
        else:
            return None
    
    # Build the path
    path = os.path.join('genres', *parts)
    return path

def create_genre_directories():
    """Create all genre directories."""
    def create_dirs(current: dict, path: str):
        if 'subgenres' in current:
            for key, value in current['subgenres'].items():
                new_path = os.path.join(path, key)
                os.makedirs(new_path, exist_ok=True)
                os.makedirs(os.path.join(new_path, 'models'), exist_ok=True)
                os.makedirs(os.path.join(new_path, 'audio'), exist_ok=True)
                if isinstance(value, dict):
                    create_dirs(value, new_path)
    
    # Create base genres directory
    os.makedirs('genres', exist_ok=True)
    
    # Create all genre directories
    for genre, data in GENRE_STRUCTURE.items():
        genre_path = os.path.join('genres', genre)
        os.makedirs(genre_path, exist_ok=True)
        os.makedirs(os.path.join(genre_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(genre_path, 'audio'), exist_ok=True)
        if 'subgenres' in data:
            create_dirs(data, genre_path)

def get_genre_name(genre_path: str) -> Optional[str]:
    """
    Get the display name for a genre path.
    
    Args:
        genre_path (str): Path in format "genre/subgenre" or "genre/subgenre/subsubgenre"
        
    Returns:
        Optional[str]: Display name if valid, None otherwise
    """
    parts = genre_path.split('/')
    current = GENRE_STRUCTURE
    
    # Navigate through the genre structure
    for part in parts:
        part = part.lower().replace(' ', '_')
        if part in current:
            current = current[part]
        elif 'subgenres' in current and part in current['subgenres']:
            current = current['subgenres'][part]
        else:
            return None
    
    # Get the name
    if isinstance(current, dict) and 'name' in current:
        return current['name']
    return current 