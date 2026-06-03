"""Language_Prompt_Selector — routes markets to native-language system prompts.

Implements design.md § Components / 4. Language Prompt Selector and
Requirements 1.1, 1.2, 1.3, 1.7.

Responsibilities
----------------

* **Market → primary language routing** (Req 1.1 / 1.7): The primary language
  of a market is the *first* entry of that market's list in
  ``_MARKET_LANGUAGES`` (the single source of truth, imported from
  :mod:`creative_agent.tools.localization_tool`). :meth:`get_primary_language`
  exposes that code and :meth:`is_native_generation_required` returns ``True``
  iff the primary language is non-English — English markets (SG, US, GB,
  EN_GLOBAL, …) bypass native-prompt switching entirely.

* **Native-language system prompts** (Req 1.2 / 1.3): A non-empty system
  prompt template — written in the target language — is provided for *every*
  value of the :class:`~creative_agent.models.Target_Language` enum. Each
  prompt instructs the LLM to think, reason, and compose copy as a native
  speaker of that language, using natural idioms and culturally conventional
  phrasing. :meth:`get_system_prompt` returns that template.

The selector itself is pure/stateless; all data is held in module-level
constants so the orchestrator and tests can use it without external resources.
"""

from __future__ import annotations

from typing import Union

from creative_agent.models import Target_Language, Target_Market

# Single source of truth for the market → languages fan-out (Req 1.1). We
# import the existing mapping rather than duplicating it so the routing here
# can never drift from the Localization_Tool's view of the world.
from creative_agent.tools.localization_tool import _MARKET_LANGUAGES

__all__ = ["LanguagePromptSelector"]

# Language code used to detect English markets (Req 1.7).
_ENGLISH_CODE: str = Target_Language.EN.value

MarketLike = Union[Target_Market, str]
LanguageLike = Union[Target_Language, str]


# ---------------------------------------------------------------------------
# Native-language system prompt templates (Req 1.2 / 1.3).
#
# Each template is written in the target language and instructs the model to
# think and compose as a native speaker. Every value of the Target_Language
# enum has a non-empty entry (Property 2 / Req 1.3). English is included so the
# selector can return a prompt for any language code, even though English
# markets do not trigger native switching (Req 1.7).
# ---------------------------------------------------------------------------
_SYSTEM_PROMPTS: dict[Target_Language, str] = {
    Target_Language.EN: (
        "You are a senior advertising copywriter and a native English speaker. "
        "Think, reason, and compose entirely in English. Write natural, "
        "persuasive ad copy that uses idiomatic English and culturally familiar "
        "phrasing for an English-speaking audience. Do not translate from "
        "another language — create the copy directly in English."
    ),
    Target_Language.FIL: (
        "Ikaw ay isang senior na copywriter sa advertising at katutubong "
        "nagsasalita ng Filipino (Tagalog). Mag-isip, mangatwiran, at "
        "lumikha ng kopya nang buong-buo sa Filipino. Gumamit ng natural na "
        "mga idyoma at kultural na angkop na pananalita para sa madlang "
        "Pilipino. Huwag magsalin mula sa ibang wika — likhain ang kopya nang "
        "direkta sa Filipino."
    ),
    Target_Language.TH: (
        "คุณเป็นนักเขียนคำโฆษณาอาวุโสและเป็นเจ้าของภาษาไทย "
        "จงคิด ใช้เหตุผล และเขียนคำโฆษณาทั้งหมดเป็นภาษาไทย "
        "ใช้สำนวนที่เป็นธรรมชาติและการใช้คำที่เหมาะสมตามวัฒนธรรมไทย "
        "อย่าแปลจากภาษาอื่น — ให้สร้างสรรค์ข้อความเป็นภาษาไทยโดยตรง"
    ),
    Target_Language.VI: (
        "Bạn là một người viết quảng cáo cấp cao và là người bản ngữ tiếng "
        "Việt. Hãy suy nghĩ, lập luận và sáng tạo toàn bộ nội dung bằng tiếng "
        "Việt. Sử dụng thành ngữ tự nhiên và cách diễn đạt phù hợp với văn hóa "
        "của người Việt. Đừng dịch từ ngôn ngữ khác — hãy viết trực tiếp bằng "
        "tiếng Việt."
    ),
    Target_Language.ID: (
        "Anda adalah seorang copywriter periklanan senior dan penutur asli "
        "Bahasa Indonesia. Berpikirlah, bernalarlah, dan susunlah seluruh "
        "naskah dalam Bahasa Indonesia. Gunakan idiom yang alami dan ungkapan "
        "yang sesuai secara budaya untuk audiens Indonesia. Jangan "
        "menerjemahkan dari bahasa lain — buatlah naskah langsung dalam "
        "Bahasa Indonesia."
    ),
    Target_Language.MS: (
        "Anda ialah seorang penulis iklan kanan dan penutur asli Bahasa "
        "Melayu. Berfikir, menaakul, dan hasilkan keseluruhan teks dalam "
        "Bahasa Melayu. Gunakan simpulan bahasa yang semula jadi dan ungkapan "
        "yang sesuai dari segi budaya untuk khalayak Malaysia. Jangan "
        "terjemah daripada bahasa lain — hasilkan teks terus dalam Bahasa "
        "Melayu."
    ),
    Target_Language.KM: (
        "អ្នកគឺជាអ្នកនិពន្ធពាណិជ្ជកម្មជាន់ខ្ពស់ និងជាអ្នកនិយាយភាសាខ្មែរដើមកំណើត។ "
        "សូមគិត វិភាគ និងបង្កើតអត្ថបទទាំងស្រុងជាភាសាខ្មែរ។ "
        "ប្រើពាក្យសម្ដីធម្មជាតិ និងការបញ្ចេញមតិដែលសមស្របតាមវប្បធម៌ខ្មែរ។ "
        "កុំបកប្រែពីភាសាផ្សេង — សូមបង្កើតអត្ថបទដោយផ្ទាល់ជាភាសាខ្មែរ។"
    ),
    Target_Language.ZH_HK: (
        "你係一位資深廣告文案撰稿人，亦係以繁體中文（香港）為母語嘅人。"
        "請全程以繁體中文思考、推理同創作文案。運用自然嘅慣用語同貼合香港"
        "文化嘅表達方式。唔好由其他語言翻譯過嚟 — 直接用繁體中文（香港）創作文案。"
    ),
    Target_Language.ZH_TW: (
        "你是一位資深廣告文案撰稿人，也是以繁體中文（台灣）為母語的人。"
        "請全程以繁體中文思考、推理並創作文案。運用自然的慣用語以及貼近台灣"
        "文化的表達方式。請勿從其他語言翻譯 — 直接以繁體中文（台灣）創作文案。"
    ),
    Target_Language.JA: (
        "あなたは日本語を母語とするシニア広告コピーライターです。"
        "すべて日本語で思考し、推論し、コピーを作成してください。"
        "自然な慣用表現と、日本の文化に即した言い回しを用いてください。"
        "他の言語から翻訳するのではなく、直接日本語でコピーを創作してください。"
    ),
    Target_Language.KO: (
        "당신은 한국어를 모국어로 사용하는 시니어 광고 카피라이터입니다. "
        "모든 사고와 추론, 카피 작성을 한국어로 진행하세요. 자연스러운 관용 "
        "표현과 한국 문화에 맞는 어법을 사용하세요. 다른 언어에서 번역하지 "
        "말고, 한국어로 직접 카피를 창작하세요."
    ),
    Target_Language.HI: (
        "आप एक वरिष्ठ विज्ञापन कॉपीराइटर हैं और हिंदी आपकी मातृभाषा है। "
        "पूरी सोच, तर्क और कॉपी रचना हिंदी में ही करें। स्वाभाविक मुहावरों "
        "और भारतीय दर्शकों के लिए सांस्कृतिक रूप से उपयुक्त अभिव्यक्ति का "
        "प्रयोग करें। किसी अन्य भाषा से अनुवाद न करें — सीधे हिंदी में कॉपी "
        "रचें।"
    ),
    Target_Language.UR: (
        "آپ ایک سینئر اشتہاری کاپی رائٹر ہیں اور اردو آپ کی مادری زبان ہے۔ "
        "اپنی پوری سوچ، استدلال اور کاپی کی تخلیق اردو میں کریں۔ فطری محاورے "
        "اور سامعین کے لیے ثقافتی طور پر موزوں اندازِ بیان استعمال کریں۔ کسی "
        "دوسری زبان سے ترجمہ نہ کریں — براہِ راست اردو میں کاپی تخلیق کریں۔"
    ),
    Target_Language.KK: (
        "Сіз — қазақ тілінде сөйлейтін ана тілді аға жарнама копирайтерісіз. "
        "Барлық ойлауды, пайымдауды және мәтін жазуды қазақ тілінде "
        "орындаңыз. Табиғи тұрақты тіркестерді және қазақ мәдениетіне сай "
        "сөз орамдарын қолданыңыз. Басқа тілден аудармаңыз — мәтінді тікелей "
        "қазақ тілінде жасаңыз."
    ),
    Target_Language.AR: (
        "أنت كاتب إعلانات محترف وناطق بالعربية الفصحى كلغة أم. فكّر واستدلّ "
        "واكتب النص الإعلاني بالكامل باللغة العربية الفصحى. استخدم التعابير "
        "الطبيعية والصياغة الملائمة ثقافيًا للجمهور العربي. لا تترجم من لغة "
        "أخرى — بل أنشئ النص مباشرةً باللغة العربية."
    ),
    Target_Language.PT_BR: (
        "Você é um redator publicitário sênior e falante nativo de português "
        "do Brasil. Pense, raciocine e crie todo o texto em português "
        "brasileiro. Use expressões idiomáticas naturais e uma linguagem "
        "culturalmente adequada ao público brasileiro. Não traduza de outro "
        "idioma — crie o texto diretamente em português do Brasil."
    ),
    Target_Language.ES: (
        "Eres un redactor publicitario sénior y hablante nativo de español. "
        "Piensa, razona y crea todo el texto en español. Utiliza expresiones "
        "idiomáticas naturales y un lenguaje culturalmente adecuado para el "
        "público hispanohablante. No traduzcas de otro idioma — crea el texto "
        "directamente en español."
    ),
    Target_Language.RU: (
        "Вы — старший рекламный копирайтер и носитель русского языка. "
        "Думайте, рассуждайте и создавайте весь текст на русском языке. "
        "Используйте естественные идиомы и культурно уместные формулировки "
        "для русскоязычной аудитории. Не переводите с другого языка — "
        "создавайте текст напрямую на русском языке."
    ),
    Target_Language.TR: (
        "Kıdemli bir reklam metin yazarı ve ana dili Türkçe olan birisiniz. "
        "Tüm düşünme, akıl yürütme ve metin oluşturma sürecini Türkçe olarak "
        "yapın. Doğal deyimler ve Türk kültürüne uygun ifadeler kullanın. "
        "Başka bir dilden çeviri yapmayın — metni doğrudan Türkçe olarak "
        "oluşturun."
    ),
    Target_Language.SW: (
        "Wewe ni mwandishi mwandamizi wa matangazo na mzungumzaji wa asili wa "
        "Kiswahili. Fikiri, tafakari, na tunga maandishi yote kwa Kiswahili. "
        "Tumia misemo ya asili na lugha inayofaa kiutamaduni kwa hadhira ya "
        "Kiswahili. Usitafsiri kutoka lugha nyingine — tunga maandishi moja "
        "kwa moja kwa Kiswahili."
    ),
}


class LanguagePromptSelector:
    """Selects native-language system prompts for Creative_Generator.

    The selector maps a market to its primary language and returns the
    matching native-language system prompt. English markets bypass native
    prompt switching entirely (Req 1.7).
    """

    #: ``market_code -> primary Target_Language code`` (design § 4). Derived
    #: from the imported :data:`_MARKET_LANGUAGES` so it stays in lock-step
    #: with the Localization_Tool. The primary language is the first entry of
    #: each market's language list (Req 1.1).
    MARKET_LANGUAGES: dict[str, str] = {
        market.value: languages[0].value
        for market, languages in _MARKET_LANGUAGES.items()
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_primary_language(self, target_market: MarketLike) -> str:
        """Return the primary language code for ``target_market`` (Req 1.1).

        The primary language is the first entry of the market's list in
        ``_MARKET_LANGUAGES``. Accepts either a :class:`Target_Market` enum
        member or its string code (e.g. ``"PH"``).

        Raises:
            KeyError: When ``target_market`` is not a known market.
        """
        market = self._coerce_market(target_market)
        return self.MARKET_LANGUAGES[market.value]

    def is_native_generation_required(self, target_market: MarketLike) -> bool:
        """Return ``True`` iff the market's primary language is non-English.

        English markets (e.g. SG, US, GB, EN_GLOBAL) return ``False`` and use
        the standard English generation flow without native-prompt switching
        (Req 1.7).
        """
        return self.get_primary_language(target_market) != _ENGLISH_CODE

    def get_system_prompt(self, target_language: LanguageLike) -> str:
        """Return the native-language system prompt for ``target_language``.

        Each prompt is written in the target language and instructs the LLM to
        think, reason, and compose as a native speaker (Req 1.2). A non-empty
        template exists for every value of the :class:`Target_Language` enum
        (Req 1.3).

        Accepts either a :class:`Target_Language` enum member or its string
        code (e.g. ``"fil"``).

        Raises:
            ValueError: When ``target_language`` is not a supported language.
        """
        language = self._coerce_language(target_language)
        return _SYSTEM_PROMPTS[language]

    # ------------------------------------------------------------------
    # Coercion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_market(target_market: MarketLike) -> Target_Market:
        if isinstance(target_market, Target_Market):
            return target_market
        try:
            return Target_Market(target_market)
        except ValueError as exc:
            raise KeyError(
                f"LanguagePromptSelector: unknown target_market {target_market!r}"
            ) from exc

    @staticmethod
    def _coerce_language(target_language: LanguageLike) -> Target_Language:
        if isinstance(target_language, Target_Language):
            return target_language
        try:
            return Target_Language(target_language)
        except ValueError as exc:
            allowed = sorted(lang.value for lang in Target_Language)
            raise ValueError(
                "LanguagePromptSelector received unsupported target language "
                f"{target_language!r}; allowed values are {allowed}"
            ) from exc
