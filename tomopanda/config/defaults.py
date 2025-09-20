"""
默认配置
"""

DEFAULT_CONFIG = {
    "detection": {
        "model": "default",
        "threshold": 0.5,
        "min_size": 10,
        "max_size": 100,
        "batch_size": 1,
        "device": "auto"
    },
    "training": {
        "epochs": 100,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "validation_split": 0.2,
        "save_interval": 10,
        "early_stopping": 20
    },
    "model": {
        "type": "se3_transformer",
        "hidden_dim": 128,
        "num_layers": 6,
        "dropout": 0.1
    },
    "data": {
        "augmentation": True,
        "normalization": True,
        "preprocessing": {
            "denoise": True,
            "enhance": True
        }
    },
    "output": {
        "format": "json",
        "save_visualization": False,
        "save_intermediate": False
    },
    "analysis": {
        "types": ["density", "distribution"],
        "bins": 50,
        "percentiles": [25, 50, 75]
    },
    "visualization": {
        "colormap": "viridis",
        "alpha": 0.8,
        "scale": 1.0,
        "dpi": 300
    },
    "system": {
        "num_workers": 4,
        "memory_limit": "8GB",
        "cache_dir": "./cache"
    }
}
