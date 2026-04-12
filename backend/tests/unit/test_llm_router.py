import pytest
from unittest.mock import patch
from reasoning.llm_router import route, TaskType, Provider, get_available_providers


def test_route_code_qa_uses_deepseek():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = "key"
        mock_settings.deepseek_api_key = "key"
        mock_settings.openrouter_api_key = "key"
        provider, model = route(TaskType.CODE_QA)
        assert provider == Provider.DEEPSEEK
        assert "coder" in model


def test_route_reasoning_uses_gemini():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = "key"
        mock_settings.deepseek_api_key = "key"
        mock_settings.openrouter_api_key = "key"
        provider, model = route(TaskType.REASONING)
        assert provider == Provider.GEMINI
        assert "gemini" in model


def test_route_falls_back_to_openrouter_when_deepseek_missing():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = "key"
        mock_settings.deepseek_api_key = ""       # DeepSeek key missing
        mock_settings.openrouter_api_key = "key"
        provider, model = route(TaskType.CODE_QA)
        assert provider == Provider.OPENROUTER


def test_route_raises_when_no_providers():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = ""
        mock_settings.deepseek_api_key = ""
        mock_settings.openrouter_api_key = ""
        with pytest.raises(RuntimeError, match="No provider available"):
            route(TaskType.CODE_QA)


def test_available_providers_reflects_keys():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = "key"
        mock_settings.deepseek_api_key = ""
        mock_settings.openrouter_api_key = "key"
        available = get_available_providers()
        assert Provider.GEMINI in available
        assert Provider.DEEPSEEK not in available
        assert Provider.OPENROUTER in available