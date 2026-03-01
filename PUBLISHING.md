# Publishing ghostfolio-agent to PyPI

Step-by-step instructions for publishing the package.

---

## Prerequisites

- Python 3.11+
- A [PyPI](https://pypi.org) account with 2FA enabled
- A PyPI API token (Account Settings > API tokens > Add token)

## 1. Create the GitHub Repository

Create an **empty** repository on GitHub (no README, no .gitignore, no license — these are already in the project):

```
Repository name: ghostfolio-agent
Description: AI-powered financial assistant for Ghostfolio portfolios
Visibility: Public
```

## 2. Remove Secrets

Before committing, make sure `.env` is deleted (it contains real API keys):

```bash
rm -f .env
```

The `.gitignore` already excludes `.env`, but delete the file to be safe.

## 3. Initialize and Push

```bash
# From the project root directory
git init
git add .
git commit -m "Initial commit: ghostfolio-agent v0.1.0"
git remote add origin git@github.com:cadechristensen/ghostfolio-agent.git
git branch -M main
git push -u origin main
```

## 4. Build the Package

```bash
pip install build
python -m build
```

This creates two files in `dist/`:
- `ghostfolio_agent-0.1.0.tar.gz` (source distribution)
- `ghostfolio_agent-0.1.0-py3-none-any.whl` (wheel)

## 5. Verify the Build

```bash
pip install twine
twine check dist/*
```

This checks that the package metadata is valid and the README renders correctly on PyPI.

## 6. Upload to PyPI

### Option A: Upload to TestPyPI first (recommended for first time)

```bash
twine upload --repository testpypi dist/*
```

Test the install:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ghostfolio-agent
ghostfolio-agent --help
```

### Option B: Upload directly to PyPI

```bash
twine upload dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: your PyPI API token (starts with `pypi-`)

## 7. Verify the Published Package

```bash
pip install ghostfolio-agent
ghostfolio-agent --help
```

---

## Releasing New Versions

1. Update the version in `pyproject.toml` and `src/ghostfolio_agent/__init__.py`
2. Commit and tag:
   ```bash
   git add .
   git commit -m "Release v0.2.0"
   git tag v0.2.0
   git push && git push --tags
   ```
3. Build and upload:
   ```bash
   rm -rf dist/
   python -m build
   twine upload dist/*
   ```

---

## Storing PyPI Credentials

To avoid entering credentials every time, create `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
```

Or use a keyring:

```bash
pip install keyring
keyring set https://upload.pypi.org/legacy/ __token__
```

---

## Troubleshooting

**"The name 'ghostfolio-agent' is already taken"**
- Someone else registered the name. Change `name` in `pyproject.toml`.

**"Invalid or non-existent authentication"**
- Make sure username is literally `__token__` (not your email)
- Make sure the API token starts with `pypi-`

**"File already exists"**
- You can't re-upload the same version. Bump the version number.

**Build fails with "No module named 'hatchling'"**
- Run `pip install hatchling` first, or use `pip install build` which handles it.
