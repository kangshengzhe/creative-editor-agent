"""Tests for Platform_Spec configuration loading."""
import pytest
from creative_agent.config import load_platform_spec
from creative_agent.models import Target_Platform, Creative_Type


class TestPlatformSpec:
    def test_google_ads_loads(self):
        spec = load_platform_spec(Target_Platform.GOOGLE_ADS)
        assert spec.platform == Target_Platform.GOOGLE_ADS

    def test_google_ads_headline_limit(self):
        spec = load_platform_spec(Target_Platform.GOOGLE_ADS)
        assert spec.char_limit(Creative_Type.HEADLINE) == 30

    def test_google_ads_description_limit(self):
        spec = load_platform_spec(Target_Platform.GOOGLE_ADS)
        assert spec.char_limit(Creative_Type.DESCRIPTION) == 90

    def test_facebook_ads_loads(self):
        spec = load_platform_spec(Target_Platform.FACEBOOK_ADS)
        assert spec.platform == Target_Platform.FACEBOOK_ADS
        assert spec.char_limit(Creative_Type.HEADLINE) == 40

    def test_tiktok_ads_loads(self):
        spec = load_platform_spec(Target_Platform.TIKTOK_ADS)
        assert spec.platform == Target_Platform.TIKTOK_ADS
        assert spec.char_limit(Creative_Type.HEADLINE) == 40

    def test_all_platforms_have_all_types(self):
        for platform in Target_Platform:
            spec = load_platform_spec(platform)
            for creative_type in Creative_Type:
                limit = spec.char_limit(creative_type)
                assert isinstance(limit, int)
                assert limit > 0

    def test_cta_limit_shorter_than_headline(self):
        spec = load_platform_spec(Target_Platform.GOOGLE_ADS)
        assert spec.char_limit(Creative_Type.CTA) < spec.char_limit(Creative_Type.HEADLINE)
