"""Tests for nanobot.audio.tts (application-level TTS service)."""

import os
import pytest

from nanobot.audio.tts import (
    EffectiveTTSConfig,
    TTSIngressError,
    resolve_tts_config,
    synthesize_speech,
)
from nanobot.config.schema import Config, TTSConfig


def _make_config(**tts_overrides) -> Config:
    config = Config()
    for k, v in tts_overrides.items():
        setattr(config.tts, k, v)
    # Ensure openai has an API key for "configured" checks
    config.providers.openai.api_key = "sk-test-key"
    return config


class TestResolveTTSConfig:
    def test_defaults(self):
        config = _make_config()
        result = resolve_tts_config(config)
        assert result.enabled is True
        assert result.provider == "openai"
        assert result.model == "tts-1"
        assert result.voice == "alloy"
        assert result.speed == 1.0
        assert result.max_char_length == 4096
        assert result.configured is True

    def test_groq_provider(self):
        config = _make_config(provider="groq")
        config.providers.groq.api_key = "gsk_test"
        result = resolve_tts_config(config)
        assert result.provider == "groq"
        assert result.model == "canopylabs/orpheus-v1-english"
        assert result.voice == "autumn"

    def test_elevenlabs_provider(self):
        config = _make_config(provider="elevenlabs")
        result = resolve_tts_config(config)
        assert result.provider == "elevenlabs"
        assert result.model == "eleven_multilingual_v2"
        assert result.voice == "Matilda"

    def test_custom_model_and_voice(self):
        config = _make_config(model="tts-1-hd", voice="nova")
        result = resolve_tts_config(config)
        assert result.model == "tts-1-hd"
        assert result.voice == "nova"

    def test_disabled(self):
        config = _make_config(enabled=False)
        result = resolve_tts_config(config)
        assert result.enabled is False

    def test_unknown_provider_falls_back(self):
        config = _make_config(provider="nonexistent")
        result = resolve_tts_config(config)
        assert result.provider == "openai"

    def test_speed_validation(self):
        config = _make_config(speed=2.0)
        result = resolve_tts_config(config)
        assert result.speed == 2.0


class TestEffectiveTTSConfig:
    def test_configured_with_api_key(self):
        cfg = EffectiveTTSConfig(
            enabled=True,
            provider="openai",
            model="tts-1",
            voice="alloy",
            speed=1.0,
            max_char_length=4096,
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
        )
        assert cfg.configured is True

    def test_not_configured_without_api_key(self):
        cfg = EffectiveTTSConfig(
            enabled=True,
            provider="openai",
            model="tts-1",
            voice="alloy",
            speed=1.0,
            max_char_length=4096,
            api_key="",
            api_base="https://api.openai.com/v1",
        )
        assert cfg.configured is False


class TestSynthesizeSpeech:
    @pytest.mark.asyncio
    async def test_raises_when_disabled(self):
        config = EffectiveTTSConfig(
            enabled=False,
            provider="openai",
            model="tts-1",
            voice="alloy",
            speed=1.0,
            max_char_length=4096,
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
        )
        with pytest.raises(TTSIngressError, match="disabled"):
            await synthesize_speech("Hello", config)

    @pytest.mark.asyncio
    async def test_raises_when_not_configured(self):
        config = EffectiveTTSConfig(
            enabled=True,
            provider="openai",
            model="tts-1",
            voice="alloy",
            speed=1.0,
            max_char_length=4096,
            api_key="",
            api_base="https://api.openai.com/v1",
        )
        with pytest.raises(TTSIngressError, match="not_configured"):
            await synthesize_speech("Hello", config)

    @pytest.mark.asyncio
    async def test_raises_on_empty_text(self):
        config = EffectiveTTSConfig(
            enabled=True,
            provider="openai",
            model="tts-1",
            voice="alloy",
            speed=1.0,
            max_char_length=4096,
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
        )
        with pytest.raises(TTSIngressError, match="empty_text"):
            await synthesize_speech("", config)

    @pytest.mark.asyncio
    async def test_raises_on_text_too_long(self):
        config = EffectiveTTSConfig(
            enabled=True,
            provider="openai",
            model="tts-1",
            voice="alloy",
            speed=1.0,
            max_char_length=10,
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
        )
        with pytest.raises(TTSIngressError, match="text_too_long"):
            await synthesize_speech("This text is definitely longer than 10 characters", config)

    @pytest.mark.asyncio
    async def test_raises_on_unknown_provider(self):
        config = EffectiveTTSConfig(
            enabled=True,
            provider="nonexistent",
            model="x",
            voice="x",
            speed=1.0,
            max_char_length=4096,
            api_key="sk-test",
            api_base="https://example.com/v1",
        )
        with pytest.raises(TTSIngressError, match="unknown_provider"):
            await synthesize_speech("Hello", config)
