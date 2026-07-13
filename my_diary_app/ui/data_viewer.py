import streamlit as st
import sys
import os
from pathlib import Path

# Fix imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from lang_diary_agentic.module_fetch_data_viewer import fetch_records_language
from lang_diary_agentic.db_handler import HandlerDairyDB
from lang_diary_agentic.vector_store import get_vector_store
from lang_diary_agentic.configs import GENERATION_DB_PATH


handler = HandlerDairyDB(GENERATION_DB_PATH)
vector_db = get_vector_store()


st.set_page_config(page_title="LinguaLog", layout="wide")

# --- 1. Sidebar Navigation ---
st.sidebar.title("📘 LinguaLog")

lang_diary_body = st.text_input("The language that you write the diary entry.", placeholder="Language code. Example: fr")

# mode = st.sidebar.radio("Navigation", ["✍️ Write Diary", "📖 History Viewer"])

# # ==========================================
# # VIEW 1: THE WRITER (Your existing logic)
# # ==========================================
# if mode == "✍️ Write Diary":
#     st.title("New Entry")
#     # ... [Insert your existing Writer code here] ...
#     # (The text_area, buttons, and graph invocation)

# ==========================================
# VIEW 2: THE DB VIEWER (New Feature)
# ==========================================
# st.title("Diary History")

if st.button("Analyze Entry"):
    # 1. Load Data
    dict_dfs = fetch_records_language(
        language_annotation=None,
        language_daiary_body=lang_diary_body,
        handler=handler,
        vector_db=vector_db
    )
    if dict_dfs is None:
        st.info("No records found yet. Go write something!")
    else:
        df_view = dict_dfs["df_merged"]
        df_diary = dict_dfs["df_diary"]
        df_error = dict_dfs["df_error"]

        # 2. Add Filters (Optional but useful)
        # Get unique languages from the dataframe
        all_langs = df_error["language_diary_text"].unique().tolist()
        selected_lang = st.multiselect("Filter by Language", all_langs, default=all_langs)
        
        # Filter the dataframe based on selection
        filtered_df = df_error[df_error["language_diary_text"].isin(selected_lang)]
        
        # 3. Render the Table
        # use_container_width=True makes it fill the screen
        # hide_index=True removes the 0,1,2 row numbers
        st.dataframe(
            filtered_df, 
            use_container_width=True, 
            hide_index=True,
        )
        
        # 4. Detailed View (Click to see full text)
        st.divider()
        st.caption("Select a row above to view details (Implementation requires st.data_editor or session state tricks, keeping it simple for now).")
        
        # Simple Alternative: Show detailed cards
        for index, row in df_diary.iterrows():
            with st.expander(f"📅 {row['date_diary']} - {row['language_source']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original:**")
                    st.info(row['diary_original'])
                with col2:
                    st.markdown("**Corrected:**")
                    st.success(row['diary_corrected'])
            # end with
        # end for
    # end if