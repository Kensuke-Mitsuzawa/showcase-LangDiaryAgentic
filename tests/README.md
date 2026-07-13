# How to run the test

```bash
export $(grep -v '^#' .env | xargs) && Mode_Deployment=cloud_api uv run python tests/test_graph.py
```

