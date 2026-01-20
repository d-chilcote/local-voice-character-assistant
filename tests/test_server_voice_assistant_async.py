import sys
import os
import asyncio
import pytest
import subprocess
from unittest.mock import MagicMock, patch, AsyncMock

# Mock dependencies before importing the module
sys.modules["llama_cpp"] = MagicMock()
sys.modules["faster_whisper"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Mock config and other modules
sys.modules["logger_config"] = MagicMock()
sys.modules["config"] = MagicMock()
sys.modules["characters"] = MagicMock()
sys.modules["utils"] = MagicMock()
sys.modules["skills.registry"] = MagicMock()
sys.modules["core.agent"] = MagicMock()
sys.modules["utils.network"] = MagicMock()

# Setup mocks for what's used at module level
mock_whisper_cls = MagicMock()
sys.modules["faster_whisper"].WhisperModel = mock_whisper_cls

mock_llama_cls = MagicMock()
sys.modules["llama_cpp"].Llama = mock_llama_cls

# Mock CHARACTERS
sys.modules["characters"].CHARACTERS = [{"name": "TestChar", "voice_native": "Alex", "voice_fallback": "Fred", "memory_file": "mem.json", "system_prompt": "prompt", "face": ":)", "id": "test"}]

# Mock cfg
mock_cfg = MagicMock()
mock_cfg.return_value.history_limit = 10
mock_cfg.return_value.llama_path = "model.gguf"
sys.modules["config"].cfg = mock_cfg

# Mock utils
sys.modules["utils"].stderr_suppressor = MagicMock()

# Mock subprocess.run for the import of server_voice_assistant
original_run = subprocess.run
mock_run = MagicMock()
mock_run.return_value.returncode = 0
subprocess.run = mock_run

try:
    import server_voice_assistant
finally:
    pass

@pytest.mark.asyncio
async def test_chat_endpoint_uses_async_subprocess():
    # Arrange
    server_voice_assistant.whisper = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello"
    server_voice_assistant.whisper.transcribe.return_value = ([mock_segment], None)

    server_voice_assistant.agent = MagicMock()
    server_voice_assistant.agent.chat.return_value = ("Response Text", [])

    subprocess.run.reset_mock()

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

                    response = await server_voice_assistant.chat_endpoint(file=mock_upload_file)

                    assert mock_async_exec.called
                    args = mock_async_exec.call_args[0]
                    assert args[0] == "say"
                    assert "Response Text" in args

                    assert mock_proc.communicate.called
                    assert response.status_code == 200

@pytest.mark.asyncio
async def test_chat_endpoint_handles_subprocess_failure():
    # Arrange
    server_voice_assistant.whisper = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Hello"
    server_voice_assistant.whisper.transcribe.return_value = ([mock_segment], None)

    server_voice_assistant.agent = MagicMock()
    server_voice_assistant.agent.chat.return_value = ("Response Text", [])

    # Mock logger
    server_voice_assistant.logger = MagicMock()

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

                response = await server_voice_assistant.chat_endpoint(file=mock_upload_file)

                assert response.status_code == 500
                server_voice_assistant.logger.error.assert_called()
                args = server_voice_assistant.logger.error.call_args[0]
                assert "Speech synthesis failed" in args[0]

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_chat_endpoint_uses_async_subprocess())
    loop.run_until_complete(test_chat_endpoint_handles_subprocess_failure())
    print("Tests passed!")
