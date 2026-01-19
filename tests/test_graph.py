from lang_diary_agentic import graph


def test_graph():
    app_graph = graph.init_graph()

    user_input = "j'ai mangé [an apple] hier soir."

    result = app_graph.invoke({"draft_text": user_input})


if __name__ == "__main__":
    test_graph()
