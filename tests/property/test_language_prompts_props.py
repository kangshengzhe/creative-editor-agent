"""Property-based tests for LanguagePromptSelector routing (task 4.2).

# Feature: creative-localization-diversity, Property 1: Language routing correctness

Validates Requirements 1.1, 1.7: the generation path chosen by the
LanguagePromptSelector matches each market's primary language — non-English
markets require native generation, English-primary markets do not.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from creative_agent.integration.language_prompts import LanguagePromptSelector
from creative_agent.models import Target_Language, Target_Market

pytestmark = pytest.mark.property

_ENGLISH = Target_Language.EN.value


# Feature: creative-localization-diversity, Property 1: Language routing correctness
@settings(max_examples=100)
@given(market=st.sampled_from(list(Target_Market)))
def test_native_generation_required_iff_primary_language_non_english(
    market: Target_Market,
) -> None:
    """`is_native_generation_required` is True iff primary language != "en".

    For every Target_Market, the native generation path is selected exactly
    when the market's primary language (the first entry in MARKET_LANGUAGES)
    is non-English. English-primary markets bypass native-prompt switching.
    """
    selector = LanguagePromptSelector()

    primary_language = selector.get_primary_language(market)
    native_required = selector.is_native_generation_required(market)

    assert native_required == (primary_language != _ENGLISH)
