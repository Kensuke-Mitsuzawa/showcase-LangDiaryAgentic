from pathlib import Path
import streamlit as st
import sys
import os

# Ensure we can import modules from 'app'
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from lang_diary_agentic.graph import init_graph


app_graph = init_graph()

st.set_page_config(page_title="LinguaLog Local", layout="wide")

st.title("📘 LinguaLog (Local LLM)")
st.caption("Running locally with Microsoft Phi-3 and ChromaDB")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. **Write** text with mixed languages: *'Je veux une [apple].'*
    2. **Retriever** looks up your past errors.
    3. **Phi-3** translates & corrects you.
    4. **Archivist** saves new errors to memory.
    """)

user_input = st.text_area("Write your diary entry:", height=150, placeholder="Example: Aujourd'hui je suis allé au [supermarket].")

if st.button("Analyze Entry"):
    if user_input:
        with st.spinner("Loading local model & thinking... (First run is slow)"):
            try:
                result = app_graph.invoke({"draft_text": user_input})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Correction")
                    st.success(result["final_response"])
                
                with col2:
                    st.subheader("🧠 Memory Context")
                    if result['retrieved_context'] != "None":
                        st.info(f"**Recall:**\n{result['retrieved_context']}")
                    else:
                        st.caption("No relevant past errors found.")
                        
                    if "None" not in result["new_errors"]:
                        st.warning(f"**New Learning Saved:**\n{result['new_errors']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text.")