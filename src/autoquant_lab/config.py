"""Credential-safe configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    mode: str = "dry-run"
    fred_api_key: str | None = None
    wrds_username: str | None = None

    @property
    def FRED_API_KEY(self) -> str | None:  # noqa: N802 - compatibility with existing setup plan naming.
        """Return the FRED API key from this config or the environment."""

        return self.fred_api_key or os.getenv("FRED_API_KEY") or None


def load_config(require_secrets: bool = False) -> Config:
    """Load environment-backed config without requiring secrets in dry-run mode."""

    _ = load_dotenv()

    fred_api_key = os.getenv("FRED_API_KEY") or None
    wrds_username = os.getenv("WRDS_USERNAME") or None

    if require_secrets and not fred_api_key:
        raise ValueError(
            "FRED_API_KEY is required for live mode. Add it to your environment or copy .env.example to .env and fill in FRED_API_KEY."
        )

    mode = "live" if require_secrets and fred_api_key else "dry-run"
    return Config(mode=mode, fred_api_key=fred_api_key, wrds_username=wrds_username)
