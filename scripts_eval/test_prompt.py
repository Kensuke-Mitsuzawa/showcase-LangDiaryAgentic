from lang_diary_agentic.clients.client_ollama import CustomOllamaServerLLM
from lang_diary_agentic.configs import Settings
from lang_diary_agentic.graph import create_compatible_chain

import time
from contextlib import contextmanager
from typing import Generator

import logging
logger = logging.getLogger(__name__)


class TimeResult:
    """A simple container to hold the duration result."""
    duration: float = 0.0


@contextmanager
def timer(label: str = "Task") -> Generator[TimeResult, None, None]:
    """
    A reusable context manager to measure code blocks.
    Yields a TimeResult object so you can access the duration 
    programmatically after the block finishes.
    
    Usage:
        with timer("Heavy Algorithm") as t:
            run_algorithm()
        print(f"Done! Result was {t.duration}")
    """
    res = TimeResult()
    start = time.perf_counter()
    try:
        yield res
    except Exception as e:
        logger.error(f"[{label}] Failed after {time.perf_counter() - start:.6f}s: {e}")
        raise RuntimeError(e)
    finally:
        res.duration = time.perf_counter() - start
        logger.debug(f"[{label}] Finished in {res.duration:.6f} seconds")    


settings = Settings()


llm = CustomOllamaServerLLM(api_url=settings.Server_API_Endpoint)

seq_llm_to_evaluate = [
    ("qwen3:8b", False),
    ()
]


test_cefr_level = [
  {
    "cefr": "A1",
    "language": "English",
    "text": "Hello, my name is Tom. I live in a small house with my cat."
  },
  {
    "cefr": "B1",
    "language": "English",
    "text": "I believe that traveling is important because it helps us understand different cultures. Last year, I visited Italy and enjoyed the local food very much."
  },
  {
    "cefr": "C1",
    "language": "English",
    "text": "The implementation of sustainable energy policies is paramount to mitigating the adverse effects of climate change. Stakeholders must collaborate effectively to ensure long-term ecological stability."
  },
  {
    "cefr": "C2",
    "language": "English",
    "text": "Notwithstanding the inherent complexities of geopolitical discourse, the prevailing consensus suggests that multilateralism remains the only viable conduit for global reconciliation. One must grapple with the subtle nuances of diplomatic etiquette to truly discern the underlying intentions."
  },
  {
    "cefr": "A1",
    "language": "German",
    "text": "Ich heiße Anna und ich komme aus Berlin. Ich trinke gern Kaffee am Morgen."
  },
  {
    "cefr": "B1",
    "language": "German",
    "text": "Obwohl das Wetter heute nicht so schön ist, möchte ich einen Spaziergang im Park machen. Ich interessiere mich sehr für die Natur und die frische Luft."
  },
  {
    "cefr": "C1",
    "language": "German",
    "text": "Es ist unumstritten, dass der technologische Fortschritt tiefgreifende Veränderungen in unserer Arbeitswelt hervorgerufen hat. Diese Entwicklung erfordert eine ständige Anpassung der beruflichen Qualifikationen."
  },
  {
    "cefr": "C2",
    "language": "German",
    "text": "In Anbetracht der prekären sozioökonomischen Lage erscheint es unabdingbar, die strukturellen Defizite des Bildungssystems einer gründlichen Analyse zu unterziehen. Nur durch eine holistische Herangehensweise lässt sich die gesellschaftliche Fragmentierung nachhaltig eindämmen."
  },
  {
    "cefr": "A1",
    "language": "French",
    "text": "Bonjour, je m'appelle Marie. J'aime manger des pommes et du fromage."
  },
  {
    "cefr": "B1",
    "language": "French",
    "text": "Si j'avais plus de temps libre, je voyagerais plus souvent à l'étranger. J'espère que je pourrai visiter le Canada l'été prochain."
  },
  {
    "cefr": "C1",
    "language": "French",
    "text": "Il convient de souligner que la préservation de la biodiversité nécessite une mobilisation citoyenne sans précédent. Les enjeux écologiques actuels nous obligent à repenser radicalement nos modes de consommation."
  },
  {
    "cefr": "C2",
    "language": "French",
    "text": "L'inexorabilité du déclin des structures traditionnelles au profit d'une virtualité omniprésente soulève des interrogations métaphysiques fondamentales. On ne saurait occulter la subtilité de cette mutation paradigmatique qui bouleverse l'essence même de l'altérité."
  }
]


# xml_schema = """\nIMPORTANT: Return the result ONLY as XML in the following structure:\n<scale>CEFR scale</scale>"""

template = [
    ("system", f"You are a skilled language teacher of English."),
    ("user", "Task: judge the language proficienfy level of the following document in the CEFR proficiency scale.\n\n{doc}")
]

llm_model_bind = llm.bind(model_name="gemma3:4b-it-qat", enable_thinking=False)

print(llm.get_available_models())

assert llm.check_connection()


chain = create_compatible_chain(template, llm_model_bind)
for _doc_obj in test_cefr_level:
    with timer(f"Level: {_doc_obj['cefr']} Lang: {_doc_obj['language']}") as t_value:
        response = chain.invoke({
            "doc": _doc_obj['text']
        })

    print(_doc_obj['cefr'], t_value.duration, response)
