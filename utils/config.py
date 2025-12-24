import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Configuration class that supports nested dictionaries and YAML loading.
    Allows both dict-style and attribute-style access.
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary"""
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, updates: Dict[str, Any]):
        """Update config with new values"""
        for key, value in updates.items():
            if isinstance(value, dict) and hasattr(self, key):
                getattr(self, key).update(value)
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default"""
        return getattr(self, key, default)
    
    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
    
    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)