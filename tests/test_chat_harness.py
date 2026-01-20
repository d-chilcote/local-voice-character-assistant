"""
Tests for the /chat web harness endpoint.

These tests verify:
1. The /chat endpoint returns HTML content
2. The harness includes expected UI elements
3. The root endpoint still returns plain text (Enchanted compatibility)
"""

import requests
import pytest

BASE_URL = "http://localhost:8888"


def test_chat_harness_returns_html():
    """Verify /chat serves the web harness as HTML."""
    print("\n--- Testing /chat harness endpoint ---")
    try:
        response = requests.get(f"{BASE_URL}/chat")
        print(f"Status: {response.status_code}")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        print("✅ Chat harness returns HTML")
    except requests.exceptions.ConnectionError:
        pytest.skip("Server not running")


def test_chat_harness_has_expected_elements():
    """Verify harness HTML contains key UI elements."""
    print("\n--- Testing harness HTML content ---")
    try:
        response = requests.get(f"{BASE_URL}/chat")
        html = response.text
        
        # Check for essential elements
        assert '<form id="chat-form"' in html, "Missing chat form"
        assert '<div id="chat">' in html, "Missing chat container"
        assert '/chat/text' in html, "Missing API endpoint reference"
        assert 'Agent Harness' in html, "Missing title"
        
        print("✅ All expected UI elements present")
    except requests.exceptions.ConnectionError:
        pytest.skip("Server not running")


def test_root_still_returns_plain_text():
    """Verify root / still returns plain text for Enchanted compatibility."""
    print("\n--- Testing root endpoint compatibility ---")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        assert "Voice Assistant Server" in response.text
        print("✅ Root endpoint returns plain text (Enchanted compatible)")
    except requests.exceptions.ConnectionError:
        pytest.skip("Server not running")


if __name__ == "__main__":
    # Note: Server must be running for these tests
    test_chat_harness_returns_html()
    test_chat_harness_has_expected_elements()
    test_root_still_returns_plain_text()
