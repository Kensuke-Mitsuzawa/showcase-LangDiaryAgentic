from pathlib import Path
import os
import pycountry
import typing

PossibleChoiceModeDeployment = typing.Literal["local", "server", "cloud_api"]
PossibleCloudLLMProvider = typing.Literal['openai', 'google']
PossibleLevelRewriting = typing.Literal['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# ---- list of language codes ----
# Iterate through all languages and filter for those that have an ISO 639 alpha_3 code
Iso693_code2natural_name = {}

Languages_Code = []
for lang in pycountry.languages:
    if hasattr(lang, 'alpha_3'):
        Languages_Code.append(lang.alpha_3)  # type: ignore
        Iso693_code2natural_name[lang.alpha_3] = lang.name  # type: ignore
    # end if
# end for
assert len(Languages_Code) > 0, "No language codes are loaded."
# ---- END: list of language codes ----