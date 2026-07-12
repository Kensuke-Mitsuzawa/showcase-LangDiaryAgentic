from lang_diary_agentic import interface
from lang_diary_agentic.configs import SettingsVariables

from datetime import datetime


def test_graph_fra_eng():
    settings = SettingsVariables()
    
    date_diary = datetime.now().isoformat()

    entry_obj = interface.DiaryEntry(
            date_diary=date_diary,
            language_source="fra",
            language_annotation="eng",
            title_diary = "Test",
            diary_original = "Je me appelle Jessica. Je suis une [girl], je suis française et je avoir [13 years old].",
            level_rewriting = "B2"
        )
    state_obj = interface.AgentState(
        diary_entry_input=entry_obj,
        parameter_config_llm = interface.ParameterConfig(
            config_translator = interface.TaskParameterConfig(
                enable_thinking = True,
            ),
            config_archivist = interface.TaskParameterConfig(
                enable_thinking = False,
            ),
            config_rewriter = interface.TaskParameterConfig(
                enable_thinking = False,
            ),
            config_reviewer = interface.TaskParameterConfig(
                enable_thinking = False,
                is_execute = False
            ),
        )
    )


    result = interface.run_pipeline(state_obj, settings=settings)

    # user_input = "Je m’appelle Jessica. Je suis une [girl], je suis française et j’ai [13 years old]. Je vais à l’école à Nice, mais j’habite à Cagnes-Sur-Mer. J’ai deux [brothers]. Le premier s’appelle Thomas, il a quatorze ans. Le second s’appelle Yann et il a neuf ans. Mon papa est italien et il est fleuriste. Ma mère est allemande et est avocate. Mes frères et moi parlons français, italien et allemand à la maison. Nous avons une grande maison avec un chien, un poisson et deux chats."
    # user_input = "Je me appelle Jessica. Je suis une [girl], je suis française et je avoir [13 years old]."

    # default_states = {
    #     "draft_text": user_input,
    #     "lang_diary_body": 'fra',
    #     "lang_annotation": 'eng',
    #     "level_rewriting": 'B2',
    #     "config_translator": graph.TaskParameterConfig(enable_thinking=True),
    #     "config_archivist": graph.TaskParameterConfig(enable_thinking=False),
    #     "config_rewriter": graph.TaskParameterConfig(enable_thinking=False),
    #     "config_reviewer": graph.TaskParameterConfig(is_execute=False),
    # }
    # result = app_graph.invoke(state_obj)
    
    # for _k, _v in result.items():
    #     print(f"{_k}: {_v}")
    # end for
# end def

# def test_graph_zho_eng():
#     app_graph = graph.init_graph()

#     user_input = "我在[diary]里写中文。我是一名[language learner]。"

#     default_states = {
#         "draft_text": user_input,
#         "lang_diary_body": 'zho',
#         "lang_annotation": 'eng',
#         "level_rewriting": 'A2',
#         "config_translator": graph.TaskParameterConfig(enable_thinking=True),
#         "config_archivist": graph.TaskParameterConfig(enable_thinking=False),
#         "config_rewriter": graph.TaskParameterConfig(enable_thinking=False),
#         "config_reviewer": graph.TaskParameterConfig(is_execute=False),
#     }
#     result = app_graph.invoke(default_states)
#     for _k, _v in result.items():
#         print(f"{_k}: {_v}")
#     # end for


if __name__ == "__main__":
    test_graph_fra_eng()
