import typing

from lingua import Language, LanguageDetectorBuilder
import logging

logger = logging.getLogger(__name__)
lang_detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
logger.debug('Loaded language detector.')

def detect_language(text: str) -> typing.Optional[str]:
    language = lang_detector.detect_language_of(text)

    if language is None:
        return None
    else:
        return language.iso_code_639_3.name.lower()