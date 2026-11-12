# Run and push (Mac/Linux)

```bash
# 1. go to the repo and unzip the package into it
cd ~/agentfail
unzip -o ~/Downloads/agentfail_dashboard.zip

# 2. environment
source .venv/bin/activate
uv pip install streamlit pandas plotly

# 3. run (opens http://localhost:8501)
streamlit run agentfail_ui.py

# 4. check tests, then commit and push on the app branch
python3 -m pytest -q
git checkout -b streamlit-app 2>/dev/null || git checkout streamlit-app
git add agentfail_ui.py docs/dashboard_context.md
git commit -m "Add Streamlit dashboard: results, failures and traces, run tab"
git push -u origin streamlit-app
```

Then open a pull request into main on GitHub. Do not commit .env or results/.

If `streamlit run` shows ModuleNotFoundError for document_loader, you ran an old file from another project. Use `streamlit run agentfail_ui.py` exactly.
