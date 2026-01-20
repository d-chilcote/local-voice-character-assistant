import sys
import os
import asyncio
import pytest
import subprocess
import importlib
from unittest.mock import MagicMock, patch, AsyncMock

# Remove top-level sys.modules hacking to prevent pollution of other tests.
# Instead, we use a fixture to patch dependencies and load the module.

@pytest.fixture
def app_module():
    """
    Fixture that mocks heavy dependencies and reloads server_voice_assistant
    to ensure a clean state without side effects on other tests.
    """
    with patch.dict(sys.modules):
        # Mock heavy dependencies
        sys.modules["llama_cpp"] = MagicMock()
        sys.modules["faster_whisper"] = MagicMock()
        # sys.modules["soundfile"] = MagicMock() # Let's use real soundfile if installed, or mock if needed.
        # Since I installed dependencies, I can skip mocking standard libs unless they are slow.

        # We need to mock things that cause side effects at import time

        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.return_value.history_limit = 10
        mock_cfg.return_value.llama_path = "model.gguf"
        # We need to patch config.cfg

        # We also need to patch subprocess.run because it's called at module level (get_voice)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0

            # We need to patch setup_logging to avoid spam
            with patch("logger_config.setup_logging"):

                # Ensure core.llm is reloaded to pick up the mocked llama_cpp
                if "core.llm" in sys.modules:
                    import core.llm
                    importlib.reload(core.llm)

                # Import or reload
                if "server_voice_assistant" in sys.modules:
                    import server_voice_assistant
                    importlib.reload(server_voice_assistant)
                else:
                    import server_voice_assistant

                yield server_voice_assistant

@pytest.mark.asyncio
async def test_chat_endpoint_uses_async_subprocess(app_module):
    # Arrange
    app_module.whisper = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello"
    app_module.whisper.transcribe.return_value = ([mock_segment], None)

    # Mock the Agent Graph
    mock_graph = AsyncMock()
    app_module.agent_graph = mock_graph

    # Mock result of ainvoke
    mock_message = MagicMock()
    mock_message.content = "Response Text"
    mock_graph.ainvoke.return_value = {"messages": [mock_message]}

    # Mock save_memory
    app_module.save_memory = AsyncMock()

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_async_exec:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")
        mock_async_exec.return_value = mock_proc

        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = b"wav_data"

            with patch("os.path.exists") as mock_exists, patch("os.remove") as mock_remove:
                mock_exists.return_value = True

                mock_upload_file = AsyncMock()
                mock_upload_file.read.return_value = b"audio_bytes"

                with patch("soundfile.read") as mock_sf_read:
                    mock_sf_read.return_value = (MagicMock(), 16000)

                    response = await app_module.chat_endpoint(file=mock_upload_file)

                    assert mock_async_exec.called
                    args = mock_async_exec.call_args[0]
                    assert args[0] == "say"
                    assert "Response Text" in args

                    assert mock_proc.communicate.called
                    assert response.status_code == 200

@pytest.mark.asyncio
async def test_chat_endpoint_handles_subprocess_failure(app_module):
    # Arrange
    app_module.whisper = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello"
    app_module.whisper.transcribe.return_value = ([mock_segment], None)

    mock_graph = AsyncMock()
    app_module.agent_graph = mock_graph
    mock_message = MagicMock()
    mock_message.content = "Response Text"
    mock_graph.ainvoke.return_value = {"messages": [mock_message]}

    # Mock save_memory
    app_module.save_memory = AsyncMock()

    # Mock logger
    app_module.logger = MagicMock()

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_async_exec:
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"Error message")
        mock_async_exec.return_value = mock_proc

        with patch("os.path.exists") as mock_exists:
             mock_exists.return_value = False # File not created

             mock_upload_file = AsyncMock()
             mock_upload_file.read.return_value = b"audio_bytes"

             with patch("soundfile.read") as mock_sf_read:
                mock_sf_read.return_value = (MagicMock(), 16000)

                response = await app_module.chat_endpoint(file=mock_upload_file)

                assert response.status_code == 500
                app_module.logger.error.assert_called()
                # Check that error was logged
                found = False
                for call in app_module.logger.error.call_args_list:
                    if "Speech synthesis failed" in str(call):
                        found = True
                        break
                assert found
