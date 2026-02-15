import os
from unittest.mock import patch
from lethe.memory.llm import LLMConfig

def test_detect_provider_gemini():
    """Test that Gemini provider is detected when GEMINI_API_KEY is present."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key", "LLM_PROVIDER": ""}):
        config = LLMConfig()
        assert config.provider == "gemini"
        # Verify default models (prefix added automatically)
        assert config.model == "gemini/gemini-2.0-flash"
        assert config.model_aux == "gemini/gemini-2.0-flash"

def test_detect_provider_gemini_explicit():
    """Test explicit Gemini provider setting."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key", "LLM_PROVIDER": "gemini"}):
        config = LLMConfig()
        assert config.provider == "gemini"
        assert config.model == "gemini/gemini-2.0-flash"
