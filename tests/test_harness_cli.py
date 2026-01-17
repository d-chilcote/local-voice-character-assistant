import pytest
from unittest.mock import patch, MagicMock
from harness_cli import run_harness
import requests

def test_harness_successful_interaction(capsys):
    """Test a successful interaction with the assistant."""
    with patch("harness_cli.input", side_effect=["Hello", "exit"]):
        with patch("harness_cli.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"speech": "Hello! How can I help you?"}
            mock_post.return_value = mock_response

            run_harness()

            captured = capsys.readouterr()
            assert "Assistant: Hello! How can I help you?" in captured.out
            assert mock_post.called

def test_harness_server_error(capsys):
    """Test handling of server errors (e.g., 500)."""
    with patch("harness_cli.input", side_effect=["Hello", "exit"]):
        with patch("harness_cli.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response

            run_harness()

            captured = capsys.readouterr()
            assert "[ERROR] Server returned 500: Internal Server Error" in captured.out

def test_harness_connection_error(capsys):
    """Test handling of connection failures."""
    with patch("harness_cli.input", side_effect=["Hello"]):
        with patch("harness_cli.requests.post", side_effect=requests.exceptions.ConnectionError("Failed to connect")):
            run_harness()

            captured = capsys.readouterr()
            assert "[ERROR] Failed to connect to agent: Failed to connect" in captured.out

def test_harness_empty_input(capsys):
    """Test that empty input is skipped and doesn't call the server."""
    # First input is empty, second is "exit" to terminate
    with patch("harness_cli.input", side_effect=["", "exit"]):
        with patch("harness_cli.requests.post") as mock_post:
            run_harness()
            assert not mock_post.called

def test_harness_quit_command(capsys):
    """Test that 'quit' command terminates the harness."""
    with patch("harness_cli.input", side_effect=["quit"]):
        with patch("harness_cli.requests.post") as mock_post:
            run_harness()
            assert not mock_post.called

def test_harness_keyboard_interrupt(capsys):
    """Test handling of KeyboardInterrupt (Ctrl+C)."""
    with patch("harness_cli.input", side_effect=KeyboardInterrupt):
        run_harness()
        # Should exit gracefully without error message (it catches KeyboardInterrupt and breaks)
        captured = capsys.readouterr()
        assert "[ERROR]" not in captured.out
