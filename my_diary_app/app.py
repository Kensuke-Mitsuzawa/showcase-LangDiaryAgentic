import logging
from pathlib import Path

from flask import Flask, render_template, request, abort

from lang_diary_agentic.graph import init_graph
from lang_diary_agentic.module_fetch_data_viewer import fetch_grammatical_errors
from lang_diary_agentic.vector_store import get_vector_store

from lang_diary_agentic.db_handler import HandlerDairyDB
from lang_diary_agentic.configs import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Initialize graph once at startup
print("Initializing Graph...")
app_graph = init_graph()

app = Flask(__name__)

DB_PATH = settings.GENERATION_DB_PATH
assert DB_PATH is not None

ChromDB_PATH = settings.ErrorVectorDB_PATH
assert ChromDB_PATH is not None


# --- VIEW The List (Table) ---
@app.route('/diary_viewer')
def index():
    handler = HandlerDairyDB(DB_PATH)
    
    seq_entries = handler.fetch_dairy_entry_language()
    if seq_entries is None:
        return render_template('diary_viewer.html', diaries=[])
    # end if
    diaries = [_r.model_dump() for _r in seq_entries]

    return render_template('diary_viewer.html', diaries=diaries)


# --- VIEW 2: Diary Detail ---
@app.route('/diary/<diary_id>')
def diary_detail(diary_id):
    handler = HandlerDairyDB(DB_PATH)
    
    entries = handler.fetch_dairy_entry_language(daiary_primary_key=diary_id)

    if entries is None:
        abort(404)
    # end if

    assert len(entries) == 1, f"The primary_id {diary_id} must be the one."

    # fetch unknown expressions
    expressions = handler.fetch_unknown_expression(daiary_primary_key=diary_id)
    if expressions is None:
        expressions = []
    # end if
    expressions = [_r.model_dump() for _r in expressions]

    diary = entries[0]
    # -----------------

    # fetch grammatical errors
    try:
        chroma_db = get_vector_store(ChromDB_PATH)
        seq_error_info = fetch_grammatical_errors(diary.primary_id, chroma_db)
        seq_error_info = [_r.model_dump() for _r in seq_error_info]
    except Exception as e:
        logger.error(f"Error fetching grammatical errors: {e}")
        seq_error_info = []
    # end try
    
    # DuckDB returns tuples. Mapping them to keys for easier HTML access (optional but recommended)
    # (Here we just use indices in the template for brevity)
    return render_template('details.html', diary=diary.model_dump(), expressions=expressions, errors=seq_error_info)


# --- 3. Routes ---
@app.route('/', methods=['GET', 'POST'])
def diary_editor():
    result = None
    error = None
    
    # Defaults for the form inputs
    form_data = {
        'draft_text': '',
        'lang_diary_body': '',
        'lang_annotation': '',
        'level_rewriting': 'B2',
        'title_diary': ''
    }

    if request.method == 'POST':
        # 1. Gather Input
        form_data = {
            "draft_text": request.form.get('draft_text'),
            "lang_diary_body": request.form.get('lang_diary_body'),
            "lang_annotation": request.form.get('lang_annotation'),
            "level_rewriting": request.form.get('level_rewriting'),
            "title_diary": request.form.get('title_diary')
        }

        if form_data['draft_text']:
            try:
                # 2. Invoke Graph
                # We simply pass the dict exactly as your logic expects
                result = app_graph.invoke(form_data)
                
            except Exception as e:
                logger.error(f"Error during invocation: {e}")
                error = str(e)
        else:
            error = "Please enter some text."

    return render_template('diary_editor.html', result=result, error=error, form=form_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)