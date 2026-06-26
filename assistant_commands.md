git add pyproject.toml fuel/ tests/
git commit -m "chore: add pyproject.toml, push fuel package"
git push origin main

pip install -e ".[dev]"
pytest tests/test_receipts.py -v