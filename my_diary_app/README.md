

```bash
export $(grep -v '^#' .env | xargs) && Mode_Deployment=cloud_api uv run python app.py
```

