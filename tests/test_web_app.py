import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import time
import tinydb
from my_diary_app.app import app, JOBS, settings
from lang_diary_agentic.db_handler import HandlerDairyDB

def test_web_app_flow():
    app.config['TESTING'] = True
    client = app.test_client()

    # 1. Access root page (editor)
    res = client.get('/')
    assert res.status_code == 200

    # 2. Submit a new diary entry for analysis
    # Simulate form data
    form_data = {
        "draft_text": "Je me appelle Jessica. Je suis une [girl], je suis française et je avoir [13 years old].",
        "lang_diary_body": "fra",
        "lang_annotation": "eng",
        "level_rewriting": "B2",
        "title_diary": "Test Web App Flow",
        # Task parameter configurations
        "config_translator.is_execute": "true",
        "config_translator.max_tokens": "512",
        "config_translator.enable_thinking": "true",
        "config_archivist.is_execute": "true",
        "config_archivist.max_tokens": "512",
        "config_archivist.enable_thinking": "false",
        "config_rewriter.is_execute": "true",
        "config_rewriter.max_tokens": "512",
        "config_rewriter.enable_thinking": "false",
        "config_reviewer.is_execute": "false",
        "config_reviewer.max_tokens": "512",
        "config_reviewer.enable_thinking": "false"
    }

    # Post request to /analyze
    response = client.post('/analyze', data=form_data)
    
    # It should redirect to /diary_viewer (status code 302)
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/diary_viewer')

    # Wait for the background thread to finish. We can check JOBS in TinyDB.
    # Get the latest job_id
    jobs = JOBS.all()
    assert len(jobs) > 0, "No background jobs registered."
    
    # Get the most recent job
    latest_job = jobs[-1]
    job_id = latest_job['job_id']
    diary_id = latest_job['diary_id']
    print(f"Tracking job {job_id} for diary {diary_id}")

    # Poll status API until it's completed or fails
    max_retries = 30
    retry_interval = 2
    completed = False

    for attempt in range(max_retries):
        status_res = client.get(f'/api/status/{job_id}')
        assert status_res.status_code == 200
        status_data = status_res.get_json()
        print(f"Poll attempt {attempt+1}: status={status_data['status']}, message={status_data['message']}")

        if status_data['status'] == 'completed':
            completed = True
            break
        elif status_data['status'] == 'error':
            pytest.fail(f"Background task failed with error: {status_data['message']}")
        
        time.sleep(retry_interval)

    assert completed, "Background job did not complete within the time limit."

    # 3. Check the diary list page
    viewer_res = client.get('/diary_viewer')
    assert viewer_res.status_code == 200
    assert b"Test Web App Flow" in viewer_res.data

    # 4. Check the detail page
    detail_res = client.get(f'/diary/{diary_id}')
    assert detail_res.status_code == 200
    assert b"Je me appelle Jessica" in detail_res.data
    # Replaced and rewritten versions should be populated
    assert b"Jessica" in detail_res.data

if __name__ == '__main__':
    test_web_app_flow()
