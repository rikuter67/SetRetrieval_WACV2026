"""
config.py - Shared configurations for the project.
"""
from typing import Dict, Any

DATASET_CONFIGS = {
    'IQON3000': {
        'num_categories': 17,
        'category_range': (1, 17),
        'category_names': {
            1: "outerwear",      # アウター（和服・ルームウェア含む）
            2: "tops",           # トップス
            3: "dresses",        # ワンピース・ドレス
            4: "pants",          # パンツ
            5: "skirts",         # スカート
            6: "shoes",          # シューズ
            7: "bags",           # バッグ
            8: "hats",           # 帽子
            9: "watches",        # 時計
            10: "accessories",   # ジュエリー・アクセサリー
            11: "fashion_goods", # ファッション雑貨（インテリア含む）
            12: "wallets",       # 財布・小物
            13: "legwear",       # レッグウェア
            14: "underwear",     # アンダーウェア・水着
            15: "beauty",        # ビューティー
            16: "glasses",       # 眼鏡
            17: "others"         # その他
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