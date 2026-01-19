from lang_diary_agentic import graph


def test_graph():
    app_graph = graph.init_graph()

    user_input = "Je m’appelle Jessica. Je suis une [girl], je suis française et j’ai [13 years old]. Je vais à l’école à Nice, mais j’habite à Cagnes-Sur-Mer. J’ai deux [brothers]. Le premier s’appelle Thomas, il a quatorze ans. Le second s’appelle Yann et il a neuf ans. Mon papa est italien et il est fleuriste. Ma mère est allemande et est avocate. Mes frères et moi parlons français, italien et allemand à la maison. Nous avons une grande maison avec un chien, un poisson et deux chats."

    result = app_graph.invoke({"draft_text": user_input})


if __name__ == "__main__":
    test_graph()
