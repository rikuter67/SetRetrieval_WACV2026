"""
config.py - Shared configurations for the project.
"""
from typing import Dict, Any

DATASET_CONFIGS = {
    'IQON3000': {
        'num_categories': 7,
        'category_range': (1, 7),
        'category_names': {
            1: "トップス系", 2: "アウター系", 3: "ボトムス系",
            4: "ワンピース・ドレス系", 5: "シューズ系",
            6: "バッグ系", 7: "アクセサリー・小物系"
        }
    },
    'DeepFurniture': {
        'num_categories': 11,
        'category_range': (1, 11),
        'category_names': {
            1: "chair", 2: "table", 3: "sofa", 4: "bed", 5: "cabinet",
            6: "lamp", 7: "bookshelf", 8: "desk", 9: "dresser",
            10: "nightstand", 11: "other_furniture"
        }
    }
}

def get_dataset_config(dataset_type: str) -> Dict[str, Any]:
    """Get configuration for a given dataset type."""
    if dataset_type not in DATASET_CONFIGS:
        print(f"[WARN] Unknown dataset type: {dataset_type}, using IQON3000 config as default.")
        dataset_type = 'IQON3000'
    return DATASET_CONFIGS[dataset_type].copy()