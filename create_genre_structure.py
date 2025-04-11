import os

GENRE_STRUCTURE = {
    'hiphop_rap': {
        'name': 'Hip-Hop / Rap',
        'subgenres': {
            'boom_bap': {'name': 'Boom Bap'},
            'trap': {
                'name': 'Trap',
                'subgenres': {
                    'hood_trap': {'name': 'Hood Trap'},
                    'trap_soul': {'name': 'Trap Soul'},
                    'rage_trap': {'name': 'Rage Trap'},
                    'latin_trap': {'name': 'Latin Trap'}
                }
            },
            'drill': {
                'name': 'Drill',
                'subgenres': {
                    'uk_drill': {'name': 'UK Drill'},
                    'ny_drill': {'name': 'NY Drill'},
                    'chicago_drill': {'name': 'Chicago Drill'}
                }
            },
            'lofi_hiphop': {'name': 'Lo-fi Hip-Hop'},
            'cloud_rap': {'name': 'Cloud Rap'},
            'phonk': {
                'name': 'Phonk',
                'subgenres': {
                    'drift_phonk': {'name': 'Drift Phonk'},
                    'memphis_phonk': {'name': 'Memphis Phonk'}
                }
            }
        }
    },
    'electronica': {
        'name': 'Electrónica',
        'subgenres': {
            'edm': {
                'name': 'EDM',
                'subgenres': {
                    'big_room': {'name': 'Big Room'},
                    'electro_house': {'name': 'Electro House'},
                    'future_house': {'name': 'Future House'}
                }
            },
            'trap_edm': {'name': 'Trap EDM / Festival Trap'},
            'dubstep': {
                'name': 'Dubstep',
                'subgenres': {
                    'riddim': {'name': 'Riddim'},
                    'brostep': {'name': 'Brostep'}
                }
            },
            'chillstep': {'name': 'Chillstep'},
            'drum_and_bass': {
                'name': 'Drum and Bass',
                'subgenres': {
                    'liquid_dnb': {'name': 'Liquid DnB'},
                    'neurofunk': {'name': 'Neurofunk'}
                }
            },
            'house': {
                'name': 'House',
                'subgenres': {
                    'deep_house': {'name': 'Deep House'},
                    'tech_house': {'name': 'Tech House'},
                    'progressive_house': {'name': 'Progressive House'}
                }
            },
            'synthwave': {'name': 'Synthwave'},
            'future_bass': {'name': 'Future Bass'},
            'jersey_club': {'name': 'Jersey Club'}
        }
    },
    'latino_mundial': {
        'name': 'Latino / Mundial',
        'subgenres': {
            'reggaeton': {
                'name': 'Reggaetón',
                'subgenres': {
                    'reggaeton_viejo': {'name': 'Reggaetón Viejo'},
                    'reggaeton_romantico': {'name': 'Reggaetón Romántico'},
                    'trapeton': {'name': 'Trapeton'}
                }
            },
            'dembow': {'name': 'Dembow'},
            'dancehall': {'name': 'Dancehall'},
            'afrobeat': {
                'name': 'Afrobeat',
                'subgenres': {
                    'afroswing': {'name': 'Afroswing'},
                    'afrotrap': {'name': 'Afrotrap'}
                }
            },
            'cumbia_electronica': {'name': 'Cumbia Electrónica'},
            'moombahton': {'name': 'Moombahton'}
        }
    },
    'otros_generos': {
        'name': 'Otros Géneros',
        'subgenres': {
            'r&b': {
                'name': 'R&B',
                'subgenres': {
                    'alternative_r&b': {'name': 'Alternative R&B'},
                    'neo_soul': {'name': 'Neo Soul'}
                }
            },
            'pop_urbano': {'name': 'Pop Urbano'},
            'experimental': {'name': 'Experimental'},
            'industrial_hiphop': {'name': 'Industrial Hip-Hop'},
            'glitch_hop': {'name': 'Glitch Hop'}
        }
    }
}

def create_genre_directories(base_dir: str = 'generos'):
    """Create the complete genre directory structure."""
    os.makedirs(base_dir, exist_ok=True)
    
    for genre_key, genre_data in GENRE_STRUCTURE.items():
        genre_dir = os.path.join(base_dir, genre_key)
        os.makedirs(os.path.join(genre_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(genre_dir, 'audio'), exist_ok=True)
        
        if 'subgenres' in genre_data:
            for subgenre_key, subgenre_data in genre_data['subgenres'].items():
                subgenre_dir = os.path.join(genre_dir, subgenre_key)
                os.makedirs(os.path.join(subgenre_dir, 'models'), exist_ok=True)
                os.makedirs(os.path.join(subgenre_dir, 'audio'), exist_ok=True)
                
                if 'subgenres' in subgenre_data:
                    for subsubgenre_key in subgenre_data['subgenres']:
                        subsubgenre_dir = os.path.join(subgenre_dir, subsubgenre_key)
                        os.makedirs(os.path.join(subsubgenre_dir, 'models'), exist_ok=True)
                        os.makedirs(os.path.join(subsubgenre_dir, 'audio'), exist_ok=True)

if __name__ == '__main__':
    create_genre_directories()
    print("✓ Estructura de géneros creada exitosamente!") 