import typing as ty
import logging
from pathlib import Path
from datetime import datetime, date
import re
import duckdb
import uuid
import threading
import tinydb
import time
from pydantic import BaseModel

from flask import Flask, render_template, abort, redirect, url_for, request, jsonify

from lang_diary_agentic.graph import init_graph, TaskParameterConfig
from lang_diary_agentic.module_fetch_data_viewer import fetch_grammatical_errors
from lang_diary_agentic.vector_store import get_vector_store

from lang_diary_agentic.models.generation_records import UnknownExpressionEntry
from lang_diary_agentic.module_post_edit import DiaryVersionManager
from lang_diary_agentic.db_handler import HandlerDairyDB
from lang_diary_agentic.configs import settings
from lang_diary_agentic.static import PossibleLevelRewriting

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


# --- IN-MEMORY JOB STORE ---
# In a real production app, use Redis/Database. For a single-user app, this dict is fine.
# Structure: { "job_id": { "status": "processing" | "completed" | "error", "result": ... } }
JOBS = tinydb.TinyDB(Path(settings.DB_BASE_DIR) / "status_db.json")

PossibleSattus = ty.Literal['completed', 'error', 'processing']
class StatusRecord(BaseModel):
    job_id: str
    status: PossibleSattus
    diary_id: str
    message: str


def parse_nested_form(form_data: ty.Dict) -> ty.Dict:
    """
    Converts flat form keys like 'config_translator[max_tokens]' 
    into nested dict {'config_translator': {'max_tokens': ...}}
    """
    nested_data = {}
    
    # 1. Regex to catch pattern: parent[child]
    pattern = re.compile(r'(\w+)\[(\w+)\]')

    for key, value in form_data.items():
        match = pattern.match(key)
        if match:
            parent, child = match.groups()
            
            if parent not in nested_data:
                nested_data[parent] = {}
            
            # Type Conversion
            if child == 'max_tokens':
                nested_data[parent][child] = int(value)
            elif child in ['is_execute', 'enable_thinking']:
                # Checkboxes send 'on' if checked. We handle 'missing' checkboxes below.
                nested_data[parent][child] = (value == 'on')

    return nested_data


def process_diary_background(job_id: str, form_data: ty.Dict, status_record: StatusRecord):
    """
    This function runs in a separate thread.
    It does the heavy LLM work and saves the result to DuckDB.
    """
    created_at = datetime.now()
    diary_date = str(date.today())
    datetime_str = created_at.isoformat()

    primary_id_DiaryEntry = f"{diary_date}_{datetime_str}"
    form_data['primary_id_DiaryEntry'] = primary_id_DiaryEntry

    status_record.diary_id = primary_id_DiaryEntry

    try:
        logger.info(f"[{job_id}] Starting background task...")
        
        # SIMULATE HEAVY WORK (Replace this with your actual app_graph.invoke)
        result = app_graph.invoke(form_data) 
        
        status_record.status = "completed"
        status_record.message = "Analysis finished successfully."
        # 3. UPDATE JOB STATUS
        JOBS.insert(status_record.model_dump())
        logger.info(f"[{job_id}] Task completed.")

    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")

        status_record.status = "error"
        status_record.message = str(e)

        JOBS.insert(status_record.model_dump())


# --- VIEW 2: Diary Detail ---
@app.route('/diary/<diary_id>/<unknown_expression_id>/delete')
def delete_unknown_expression(diary_id, unknown_expression_id, methods=['GET']):
    handler = HandlerDairyDB(DB_PATH)

    entries = handler.fetch_dairy_entry_language(daiary_primary_key=diary_id)
    assert entries is not None
    current_version = entries[0].current_version

    unknown_expressions = handler.fetch_unknown_expression(daiary_primary_key=diary_id)
    target_delete = [_r for _r in unknown_expressions if _r.primary_id == unknown_expression_id]

    history_record = DiaryVersionManager().create_history_record_expression_filed(
        diary_id=diary_id,
        current_version=current_version,
        operation="delete",
        primary_id=unknown_expression_id,
        expression_original=target_delete[0].expression,
        expression_translation=target_delete[0].expression_translation
    )
    assert history_record is not None

    # delete the unknown expression record
    db_con = duckdb.connect(DB_PATH)
    db_con.execute("DELETE from unknown_expressions WHERE primary_id = ?", (unknown_expression_id, ))
    db_con.commit()
    db_con.close()

    handler.save_diary_version_history(history_record)

    return redirect(url_for('diary_detail', diary_id=diary_id))


@app.route('/diary/<diary_id>/add_expression', methods=['POST'])
def add_expression_logic(diary_id):
    handler = HandlerDairyDB(DB_PATH)

    # 1. Get the data from the form
    expression = request.form.get('expression_original')
    translation = request.form.get('expression_translation')

    entries = handler.fetch_dairy_entry_language(daiary_primary_key=diary_id)
    assert entries is not None
    current_version = entries[0].current_version

    unknown_exp_entry = UnknownExpressionEntry(
        expression=expression,
        expression_translation=translation,
        span_original=(-1, -1),
        span_translation=(-1, -1),
        language_source=entries[0].language_source,
        language_annotation=entries[0].language_annotation,
        created_at=datetime.now(),
        primary_id_DiaryEntry=entries[0].primary_id
    )

    history_record = DiaryVersionManager().create_history_record_expression_filed(
        diary_id=diary_id,
        current_version=current_version,
        operation="add",
        primary_id=unknown_exp_entry.primary_id,
        expression_original=expression,
        expression_translation=translation
    )
    assert history_record is not None


    handler.save_unknown_expression(unknown_exp_entry)
    handler.save_diary_version_history(history_record)
    
    # 
    db_con = duckdb.connect(DB_PATH)
    db_con.execute("UPDATE diary_entries SET current_version = ? WHERE primary_id = ?", (history_record.version_to, diary_id))
    db_con.commit()
    db_con.close()

    # 3. Redirect back to the diary entry view
    return redirect(url_for('diary_detail', diary_id=diary_id))


@app.route('/diary/<diary_id>/update_text', methods=['POST'])
def update_diary_text(diary_id):
    handler = HandlerDairyDB(DB_PATH)

    # 1. Get the data from the form
    update_rewriting = request.form.get('update_rewriting', None)
    if update_rewriting is None:
        return redirect(url_for('diary_detail', diary_id=diary_id))
    # end if

    entries = handler.fetch_dairy_entry_language(daiary_primary_key=diary_id)
    assert entries is not None
    
    current_version = entries[0].current_version

    history_record = DiaryVersionManager().create_history_record_text_filed(
        diary_id=diary_id,
        current_version=current_version,
        field_name="diary_rewritten",
        old_text=entries[0].diary_rewritten,
        new_text=update_rewriting
    )
    if history_record is None:
        return redirect(url_for('diary_detail', diary_id=diary_id))
    # end if

    # 
    db_con = duckdb.connect(DB_PATH)
    db_con.execute("UPDATE diary_entries SET diary_rewritten = ?, current_version = ? WHERE primary_id = ?", (update_rewriting, history_record.version_to, diary_id))
    db_con.commit()
    db_con.close()

    handler.save_diary_version_history(history_record)
    
    # 3. Redirect back to the diary entry view
    return redirect(url_for('diary_detail', diary_id=diary_id))


@app.route('/diary/<diary_id>')
def diary_detail(diary_id, methods=['GET']):
    handler = HandlerDairyDB(DB_PATH)
    
    entries = handler.fetch_dairy_entry_language(daiary_primary_key=diary_id)

    if entries is None:
        abort(404)
    # end if

    assert len(entries) == 1, f"The primary_id {diary_id} must be the one."

    # ---- fetch unknown expressions ----
    expressions = handler.fetch_unknown_expression(daiary_primary_key=diary_id)
    if expressions is None:
        expressions = []
    # end if
    expressions = [_r.model_dump() for _r in expressions]

    diary = entries[0]
    # ---- END: fetch unknown expressions ----

    # ---- fetch grammatical errors ----
    try:
        chroma_db = get_vector_store(ChromDB_PATH)
        seq_error_info = fetch_grammatical_errors(diary.primary_id, chroma_db)
        seq_error_info = [_r.model_dump() for _r in seq_error_info]
    except Exception as e:
        logger.error(f"Error fetching grammatical errors: {e}")
        seq_error_info = []
    # end try
    # ---- END: fetch grammatical errors ----

    # DuckDB returns tuples. Mapping them to keys for easier HTML access (optional but recommended)
    # (Here we just use indices in the template for brevity)
    form_data = {
        'input_rewriting': diary.diary_rewritten
    }
    return render_template('details.html', 
                           diary=diary.model_dump(), 
                           expressions=expressions, 
                           errors=seq_error_info,
                           form=form_data)


# --- VIEW The List (Table) ---

@app.route('/diary_viewer')
def diary_viewer():
    handler = HandlerDairyDB(DB_PATH)
    
    seq_entries = handler.fetch_dairy_entry_language(is_show_only=True)
    if seq_entries is None:
        return render_template('diary_viewer.html', diaries=[])
    # end if
    diaries = [_r.model_dump() for _r in seq_entries]
    for _obj in diaries:
        q = tinydb.Query()
        output = JOBS.search(q.diary_id == _obj['primary_id'])
        if output is None or output == []:
            _emoji = "❓"
            _obj['status'] = _emoji
        else:
            r = StatusRecord(**output[0])
            if r.status == "completed":
                _emoji = "✅"
            elif r.status == "error":
                _emoji = "❌"
            elif r.status == "processing":
                _emoji = "⏳"
            else:
                _emoji = "❓" 
            # end if
            _obj['status'] = _emoji
    # end for

    return render_template('diary_viewer.html', diaries=diaries)


@app.route('/diary_viewer/delete/<diary_id>', methods=["GET"])
def make_diary_invaid(diary_id: str):
    handler = HandlerDairyDB(DB_PATH)
    
    seq_entries = handler.fetch_dairy_entry_language(is_show_only=True, daiary_primary_key=diary_id)
    if seq_entries is None:
        return redirect(url_for('diary_viewer'))
    # end if
    assert len(seq_entries) == 1

    conn = duckdb.connect(DB_PATH)
    conn.execute("UPDATE diary_entries SET is_show = ? WHERE primary_id = ?", (False, diary_id))

    return redirect(url_for('diary_viewer'))


@app.route('/api/status/<job_id>')
def api_check_status(job_id):
    """API endpoint for JavaScript to poll."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "unknown"}), 404
    return jsonify(job)


@app.route('/api/status/<diary_id>')
def api_dialy_status(diary_id):
    """API endpoint for JavaScript to poll."""
    q = tinydb.Query()
    output = JOBS.search(q.diary_id == diary_id)
    r = StatusRecord(**output)

    if r.status == "completed":
        _emoji = "✅"
    elif r.status == "error":
        _emoji = "❌"
    elif r.status == "processing":
        _emoji = "⏳"
    else:
        _emoji = "❓"
    
    return jsonify({
        "code_status": r.status,
        "emoji": _emoji
    })




# --- 3. Routes ---


@app.route('/analyze', methods=['POST'])
def analyze_entry():
    form_data = request.form.to_dict() # Convert ImmutableDict to standard dict
    assert "draft_text" in form_data
    assert "lang_diary_body" in form_data
    assert "lang_annotation" in form_data
    assert "level_rewriting" in form_data
    assert "title_diary" in form_data
    
    param_obj = {
        "draft_text": request.form.get('draft_text'),
        "lang_diary_body": request.form.get('lang_diary_body'),
        "lang_annotation": request.form.get('lang_annotation'),
        "level_rewriting": request.form.get('level_rewriting'),
        "title_diary": request.form.get('title_diary')
    }
    
    parsed_dict = parse_nested_form(request.form)
    form_config_llm_exec = {
        "config_translator": TaskParameterConfig(
            is_execute=True,
            max_tokens=parsed_dict.get('config_translator', {}).get('max_tokens', 512),
            enable_thinking=parsed_dict.get('config_translator', {}).get('enable_thinking', False),
        ),
        "config_archivist": TaskParameterConfig(
            is_execute=True,
            max_tokens=parsed_dict.get('config_archivist', {}).get('max_tokens', 512),
            enable_thinking=parsed_dict.get('config_archivist', {}).get('enable_thinking', False),
        ),
        "config_rewriter": TaskParameterConfig(
            is_execute=parsed_dict.get('config_rewriter', {}).get('is_execute', False),
            max_tokens=parsed_dict.get('config_rewriter', {}).get('max_tokens', 512),
            enable_thinking=parsed_dict.get('config_rewriter', {}).get('enable_thinking', False),
        ),
        "config_reviewer": TaskParameterConfig(
            is_execute=parsed_dict.get('config_reviewer', {}).get('is_execute', False),
            max_tokens=parsed_dict.get('config_reviewer', {}).get('max_tokens', 512),
            enable_thinking=parsed_dict.get('config_reviewer', {}).get('enable_thinking', False),
        )
    }

    if form_data['draft_text']:
        try:
            param_obj.update(form_config_llm_exec)
        except Exception as e:
            logger.error(f"Error during invocation: {e}")
            error = str(e)
            return render_template('diary_editor.html', error=error, form=form_data, possible_rewriting_level=ty.get_args(PossibleLevelRewriting))
    else:
        error = "Please enter some text."
        return render_template('diary_editor.html', error=error, form=form_data, possible_rewriting_level=ty.get_args(PossibleLevelRewriting))

    # end if

    # 2. Generate a unique Job ID
    job_id = str(uuid.uuid4())
    
    # 3. Set initial status
    r = StatusRecord(
        job_id=job_id,
        status='processing',
        diary_id='',
        message='')
    JOBS.insert(r.model_dump())

    # 4. Start the background thread
    # daemon=True means this thread dies if the main app crashes (good for cleanup)
    thread = threading.Thread(target=process_diary_background, args=(job_id, param_obj, r), daemon=True)
    thread.start()
    
    time.sleep(10)
    # 5. IMMEDIATELY redirect user to the Status Page
    return redirect(url_for('diary_viewer'))


@app.route('/', methods=['GET'])
def diary_editor():
    result = None
    error = None

    handler = HandlerDairyDB(DB_PATH)
    handler.init_db()
    
    # Defaults for the form inputs
    form_data = {
        'draft_text': '',
        'lang_diary_body': '',
        'lang_annotation': '',
        'level_rewriting': 'B2',
        'title_diary': ''
    }

    return render_template('diary_editor.html', 
                           error=error, 
                           form=form_data, 
                           possible_rewriting_level=ty.get_args(PossibleLevelRewriting))


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')