# document_aligner/src/reporting/markdown_writer.py

from pathlib import Path
from typing import List, Dict, Any

ContentItem = Dict[str, Any]

def save_to_markdown(content: List[ContentItem], filepath: Path) -> None:
    """
    Saves the processed document content to a Markdown file.

    Each content item is written as a separate paragraph.

    Args:
        content: A list of structured content items.
        filepath: The path to the output Markdown file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in content:
            # Add a simple marker for headings for better readability
            if item['type'] in {'title', 'sectionHeading', 'subheading'}:
                f.write(f"## {item['text']}\n\n")
            else:
                f.write(f"{item['text']}\n\n")