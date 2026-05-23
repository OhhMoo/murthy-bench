import json
import os
from pathlib import Path

_CONFIG_PATH = Path.home() / ".longevity" / "config.json"

_DEFAULTS: dict = {
    "hf.token": None,
    "openai.api_key": None,
    "anthropic.api_key": None,
    "eval.concurrency": 4,
    "eval.budget": 5,
}

_ENV_OVERRIDES: dict = {
    "hf.token": "HF_TOKEN",
    "openai.api_key": "OPENAI_API_KEY",
    "anthropic.api_key": "ANTHROPIC_API_KEY",
}

_PROVIDER_KEY_MAP: dict = {
    "hf": "hf.token",
    "openai": "openai.api_key",
    "anthropic": "anthropic.api_key",
    "endpoint": None,  # api key passed directly via --api-key flag
}


def _load_file() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_file(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(data, indent=2))


def get(key: str):
    env_var = _ENV_OVERRIDES.get(key)
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val
    file_data = _load_file()
    if key in file_data:
        return file_data[key]
    return _DEFAULTS.get(key)


def set_value(key: str, value) -> None:
    data = _load_file()
    data[key] = value
    _save_file(data)


def all_values() -> dict:
    data = _load_file()
    result = dict(_DEFAULTS)
    result.update(data)
    for key, env_var in _ENV_OVERRIDES.items():
        val = os.environ.get(env_var)
        if val:
            result[key] = val
    return result


def provider_api_key(provider: str) -> str | None:
    config_key = _PROVIDER_KEY_MAP.get(provider)
    if config_key is None:
        return None
    return get(config_key)


def provider_preflight(provider: str, api_key: str | None = None) -> str | None:
    """Return an error message if the provider is missing credentials, else None."""
    if api_key:
        return None
    resolved = provider_api_key(provider)
    if not resolved:
        env_hint = ""
        config_key = _PROVIDER_KEY_MAP.get(provider)
        if config_key:
            env_var = _ENV_OVERRIDES.get(config_key, "")
            env_hint = f" (set {env_var} or run: longevity config set {config_key} <value>)"
        return f"No API key found for provider '{provider}'{env_hint}"
    return None
