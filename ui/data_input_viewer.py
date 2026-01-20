from pathlib import Path
import streamlit as st
import sys
import logging
import json

# Ensure we can import modules from 'app'
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from lang_diary_agentic.graph import init_graph


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


app_graph = init_graph()

st.set_page_config(page_title="LinguaLog Local", layout="wide")

st.title("📘 LinguaLog")
st.caption("Your personal diary-writing tutor for the language learning.")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **Write** text with mixed languages: *'Je veux une [apple].'*
    2. **Retriever** looks up your past errors.
    3. **LLMs** translates & corrects you.
    4. **Archivist** saves new errors to memory.
    """)



user_input = st.text_area("Write your diary entry:", height=150, placeholder="Example: Je m’appelle Jessica. Je suis une [girl], je suis française et j’ai [13 years old]. Je vais à l’école à Nice, mais j’habite à Cagnes-Sur-Mer. J’ai deux [brothers].")
st.caption("Example Text: Je me appelle Jessica. Je suis [a girl], je suis français et je avoir [13 years old].")
lang_diary_body = st.text_input("The language that you write the diary entry.", placeholder="Language code. Example: fr")
lang_annotation = st.text_input("The language that you use for the annotation [LANG].", placeholder="Language code. Example: en")
lang_level_rewriting = st.text_input("Preferred level for the rewriting. Ex. B1, B2, C1", placeholder="B2", value="B2")

if st.button("Analyze Entry"):
    if user_input:
        with st.spinner("Loading local model & thinking... (First run is slow)"):
            try:
                result = app_graph.invoke({
                    "draft_text": user_input,
                    "lang_diary_body": lang_diary_body,
                    "lang_annotation": lang_annotation,
                    "level_rewriting": lang_level_rewriting
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # showing corrections
                    st.subheader("📝 Correction")
                    st.success(result["final_response"])

                    st.subheader(f"📝 Suggestion in {lang_level_rewriting} level")
                    st.success(result["suggestion_response"])                    

                    # showing the unknown expressions
                    st.subheader("❓unknown expression")
                    _seq_unkown_expressions = result["unkown_expressions"]
                    for _expression_obj in _seq_unkown_expressions:
                        _expression_unk = _expression_obj['expression_original']
                        _expression_translation = _expression_obj['expression_translation']
                        st.success(f'[{_expression_unk}] -> {_expression_translation}')
                    # end for

                with col2:
                    st.subheader("🧠 Memory Context")
                    if result['retrieved_context'] != "None":
                        st.info(f"**Recall:**\n{result['retrieved_context']}")
                    else:
                        st.caption("No relevant past errors found.")
                    # end if
                    if "None" not in result["new_errors"]:
                        st.warning("**New Learning Saved:**\n")
                        _obj_error_analysis = json.loads(result['new_errors'])
                        for _entry in _obj_error_analysis:
                            _document_output = ""
                            _document_output += f"Error-category: {_entry['category']}\n\n"
                            _document_output += f"Error-type: {_entry['error_rule']}\n\n"
                            _document_output += f"Correction: {_entry['correction']}\n\n"
                            _document_output += f"Input: {_entry['example_phrase']}\n\n"
                            st.warning(f"{_document_output}")                            
                        # end for
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text.")