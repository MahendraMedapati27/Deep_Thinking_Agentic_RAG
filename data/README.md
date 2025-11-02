# Data Directory

Place your documents here to be processed by the Deep-Thinking RAG system.

## Supported File Types

- `.txt` - Plain text files (any kind of text document)

## Example Documents

You can use:
- Financial filings (10-K, 10-Q, etc.)
- Research papers
- Technical documentation
- News articles
- Legal documents
- Educational content
- Corporate reports
- Any text-based document

## Processing

Documents will be:
1. Automatically chunked with section metadata
2. Embedded using local sentence-transformers model
3. Indexed in the vector store for search

## Getting Started

1. Add your document to this folder
2. Run the application
3. Upload the document through the UI or the CLI will auto-detect it

