# EDI-AI-System Web UI

Web interface for browsing EDI resources and accessing AI-powered Q&A.

## Quick Start

### Prerequisites

- Python 3.8+
- SQLite database at `data/processed/summaries.sqlite`
- Required Python packages (see `requirements.txt`)

### Installation

Install dependencies (if not already installed):

```bash
pip install -r requirements.txt
```

### Running the Web App

From the project root directory, run:

```bash
python -m web.app
```

Or using uvicorn directly:

```bash
uvicorn web.app:app --reload
```

The app will start on `http://localhost:8000`

### Testing Database Connection

Verify database connectivity:

```bash
python -m web.db_check
```

This will print 5 sample resources and summary statistics.

## Pages

- **Home** (`/`): Introduction to EDI and navigation to main features
- **Resources** (`/resources`): Browse all Included resources with search
- **Resource Detail** (`/resource/{id}`): Preview and AI summary for a specific resource
- **Q&A** (`/qa`): Placeholder for future RAG Q&A integration

## Features

### Resource Preview

- **PDFs**: Embedded PDF viewer for local PDF files
- **HTML**: Clean text preview from extracted text files
- **Fallback**: External link to original resource

### Search

Search resources by title or author on the Resources page.

### Security

- PDF files are served securely from `data/raw/downloads` only
- Path traversal protection ensures files cannot be accessed outside allowed directories

## Development

### Project Structure

```
web/
├── app.py              # FastAPI application
├── db_check.py         # Database connectivity test
├── templates/          # Jinja2 HTML templates
│   ├── base.html
│   ├── home.html
│   ├── resources.html
│   ├── resource_detail.html
│   └── qa.html
├── static/             # Static assets
│   └── style.css
└── README.md
```

### Configuration

Database path is configured in `app.py`:

```python
DB_PATH = PROJECT_ROOT / "data" / "processed" / "summaries.sqlite"
```

To use a different database, modify this path.

## Known Limitations

1. **Iframe Blocking**: Many external websites block iframe embedding. HTML resources default to text preview instead.
2. **Q&A Placeholder**: The Q&A page is currently a placeholder. RAG integration coming soon.
3. **Missing Text Files**: Resources without extracted text files will show "Preview not available".

## Troubleshooting

### Database Not Found

Ensure `data/processed/summaries.sqlite` exists. Run the summarisation script if needed.

### PDFs Not Loading

- Verify PDF files exist in `data/raw/downloads/`
- Check that `raw_path` in database points to valid files
- Ensure file permissions allow reading

### Static Files Not Loading

Ensure the `web/static/` directory exists and contains `style.css`.
