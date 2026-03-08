# tests/conftest.py
# Global fixture to mock required environment variables for all tests
# Prevents config.py from raising EnvironmentError during import

import pytest
import os

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    # Mock all required keys with dummy values
    monkeypatch.setenv('ALPACA_API_KEY', 'dummy_key')
    monkeypatch.setenv('ALPACA_API_SECRET', 'dummy_secret')
    monkeypatch.setenv('POLYGON_API_KEY', 'dummy_polygon')
    monkeypatch.setenv('TIINGO_API_KEY', 'dummy_tiingo')
    monkeypatch.setenv('NEWS_API_KEY', 'dummy_news')
    # Add any other env vars your code might check
