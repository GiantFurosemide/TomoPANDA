"""
配置管理器
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .defaults import DEFAULT_CONFIG


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("tomopanda_config.json")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.suffix == '.yaml' or self.config_path.suffix == '.yml':
                        return yaml.safe_load(f)
                    else:
                        return json.load(f)
            except Exception as e:
                print(f"警告: 无法加载配置文件 {self.config_path}: {e}")
                print("使用默认配置")
        
        return DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """保存配置文件"""
        if config is None:
            config = self.config
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            if self.config_path.suffix == '.yaml' or self.config_path.suffix == '.yml':
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
