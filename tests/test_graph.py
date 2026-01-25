from lang_diary_agentic import graph



def test_graph_fra_eng():
    app_graph = graph.init_graph()

    # user_input = "Je m’appelle Jessica. Je suis une [girl], je suis française et j’ai [13 years old]. Je vais à l’école à Nice, mais j’habite à Cagnes-Sur-Mer. J’ai deux [brothers]. Le premier s’appelle Thomas, il a quatorze ans. Le second s’appelle Yann et il a neuf ans. Mon papa est italien et il est fleuriste. Ma mère est allemande et est avocate. Mes frères et moi parlons français, italien et allemand à la maison. Nous avons une grande maison avec un chien, un poisson et deux chats."
    user_input = "Je me appelle Jessica. Je suis une [girl], je suis française et je avoir [13 years old]."

    default_states = {
        "draft_text": user_input,
        "lang_diary_body": 'fra',
        "lang_annotation": 'eng',
        "level_rewriting": 'B2',
        "config_translator": graph.TaskParameterConfig(enable_thinking=True),
        "config_archivist": graph.TaskParameterConfig(enable_thinking=False),
        "config_rewriter": graph.TaskParameterConfig(enable_thinking=False),
        "config_reviewer": graph.TaskParameterConfig(is_execute=False),
    }
    result = app_graph.invoke(default_states)
    for _k, _v in result.items():
        print(f"{_k}: {_v}")
    # end for
# end def

def test_graph_zho_eng():
    app_graph = graph.init_graph()

    user_input = "我在[diary]里写中文。我是一名[language learner]。"

    default_states = {
        "draft_text": user_input,
        "lang_diary_body": 'zho',
        "lang_annotation": 'eng',
        "level_rewriting": 'A2',
        "config_translator": graph.TaskParameterConfig(enable_thinking=True),
        "config_archivist": graph.TaskParameterConfig(enable_thinking=False),
        "config_rewriter": graph.TaskParameterConfig(enable_thinking=False),
        "config_reviewer": graph.TaskParameterConfig(is_execute=False),
    }
    result = app_graph.invoke(default_states)
    for _k, _v in result.items():
        print(f"{_k}: {_v}")
    # end for


if __name__ == "__main__":
    test_graph_fra_eng()
