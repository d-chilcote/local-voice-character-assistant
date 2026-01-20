import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Mock heavy dependencies
sys.modules["faster_whisper"] = MagicMock()
sys.modules["llama_cpp"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Setup specific mocks
mock_whisper = MagicMock()
sys.modules["faster_whisper"].WhisperModel.return_value = mock_whisper

mock_llama = MagicMock()
sys.modules["llama_cpp"].Llama.return_value = mock_llama

mock_sf = MagicMock()
sys.modules["soundfile"].read.return_value = (MagicMock(), 16000)

# Mock config and other utils if needed
# Assuming they are lightweight or we can let them run
# We might need to mock 'characters' if it does heavy stuff, but it looks like just a list.

import pytest
import asyncio
import os

# We need to mock os.path.exists and os.remove and open
# But first let's import the module.
# We need to patch 'server_voice_assistant.subprocess.run' which is used in 'get_voice' at module level
# Also 'server_voice_assistant.setup_logging'
# And 'server_voice_assistant.Agent'

with patch("subprocess.run") as mock_run, \
     patch("core.agent.Agent") as mock_agent_cls, \
     patch("logger_config.setup_logging"):

    # Configure mock run for get_voice
    mock_run.return_value.returncode = 0

    # Configure mock agent
    mock_agent_instance = MagicMock()
    mock_agent_instance.chat.return_value = ("Hello world", [])
    mock_agent_cls.return_value = mock_agent_instance

    import server_voice_assistant

@pytest.mark.asyncio
async def test_chat_endpoint_uses_async_subprocess():
    # Setup
    server_voice_assistant.whisper.transcribe.return_value = ([MagicMock(text="Hello")], None)
    server_voice_assistant.agent.chat.return_value = ("Response text", [])

    # Mock UploadFile
    mock_file = AsyncMock()
    mock_file.read.return_value = b"fake audio data"

    # Mock asyncio.create_subprocess_exec
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
         patch("builtins.open", new_callable=MagicMock) as mock_open, \
         patch("os.path.exists", return_value=False), \
         patch("os.remove"):

        # Configure the process mock
        mock_proc = AsyncMock()
        mock_proc.wait.return_value = None
        mock_exec.return_value = mock_proc

        # Configure file read
        mock_file_handle = MagicMock()
        mock_file_handle.read.return_value = b"wav data"
        mock_open.return_value.__enter__.return_value = mock_file_handle

        # Call the endpoint
        response = await server_voice_assistant.chat_endpoint(file=mock_file)

        # Verify asyncio.create_subprocess_exec was called
        assert mock_exec.called
        args = mock_exec.call_args
        assert args[0][0] == "say"
        assert "Response text" in args[0]

        # Verify we waited for the process
        assert mock_proc.wait.called

        # Verify response
        assert response.body == b"wav data"

if __name__ == "__main__":
    # Manually run the test function if run as script
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_chat_endpoint_uses_async_subprocess())
    print("Test passed!")
