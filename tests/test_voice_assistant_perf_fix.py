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
     patch("core.agent_graph.create_agent_graph") as mock_create_graph, \
     patch("logger_config.setup_logging"):

    # Configure mock run for get_voice
    mock_run.return_value.returncode = 0

    # Configure mock agent graph
    mock_agent_graph = AsyncMock()
    mock_create_graph.return_value = mock_agent_graph

    import server_voice_assistant

@pytest.mark.asyncio
async def test_chat_endpoint_uses_async_subprocess():
    # Setup
    # Mock agent_graph invoke result
    server_voice_assistant.agent_graph.ainvoke = AsyncMock()
    server_voice_assistant.agent_graph.ainvoke.return_value = {
        "messages": [MagicMock(content="Response text")]
    }

    # Mock UploadFile
    mock_file = AsyncMock()
    mock_file.read.return_value = b"fake audio data"

    # Mock asyncio.create_subprocess_exec
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
         patch("builtins.open", new_callable=MagicMock) as mock_open, \
         patch("server_voice_assistant.os.path.exists", return_value=True), \
         patch("server_voice_assistant.os.remove"):

        # Configure the process mock
        mock_proc = AsyncMock()
        mock_proc.wait.return_value = None
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
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

        # Verify we communicated with the process
        assert mock_proc.communicate.called

        # Verify response
        assert response.body == b"wav data"

if __name__ == "__main__":
    # Manually run the test function if run as script
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_chat_endpoint_uses_async_subprocess())
    print("Test passed!")
