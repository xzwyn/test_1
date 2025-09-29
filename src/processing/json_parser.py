# document_aligner/src/processing/json_parser.py

import json
from pathlib import Path
from typing import List, Dict, Any

import config

# A type alias for our structured content for clarity
ContentItem = Dict[str, Any]

def process_document_json(filepath: Path) -> List[ContentItem]:
    """
    Reads and processes an Azure Document Intelligence JSON file.

    This function extracts paragraphs, identifies their roles (e.g., title, heading, paragraph),
    filters out ignored content like headers/footers, and stitches together paragraphs
    that may have been split across pages.

    Args:
        filepath: The path to the input JSON file.

    Returns:
        A list of dictionaries, where each dictionary represents a
        structured content segment (e.g., a heading or a full paragraph).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"The file '{filepath}' was not found.")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        full_text_content = data['analyzeResult']['content']
        raw_paragraphs = data['analyzeResult']['paragraphs']
        pages = data['analyzeResult']['pages']
    except KeyError as e:
        raise ValueError(f"JSON file '{filepath}' is missing expected key: {e}") from e
    
    # Create a quick lookup for page number by span offset
    page_lookup = {}
    for page in pages:
        for span in page['spans']:
            for i in range(span['offset'], span['offset'] + span['length']):
                page_lookup[i] = page['pageNumber']

    # --- First Pass: Extract and filter raw segments from JSON ---
    initial_segments = []
    for p in raw_paragraphs:
        role = p.get('role', 'paragraph') # Default to 'paragraph' if role is missing

        if role in config.IGNORED_ROLES:
            continue

        if p['spans']:
            offset = p['spans'][0]['offset']
            length = p['spans'][0]['length']
            text = full_text_content[offset : offset + length].strip()
            page_number = page_lookup.get(offset, 0)

            if text:
                initial_segments.append({'text': text, 'type': role, 'page': page_number})

    # --- Second Pass: Stitch broken paragraphs ---
    # This logic combines segments that are not structural elements (like headings)
    # and do not end with terminal punctuation, indicating they were split.
    final_content: List[ContentItem] = []
    stitched_text = ""
    current_page = 0

    for i, segment in enumerate(initial_segments):
        is_structural = segment['type'] in config.STRUCTURAL_ROLES
        text_ends_sentence = segment['text'].endswith(('.', '!', '?', ':', '•'))

        if stitched_text == "": # Start of a new segment
             stitched_text = segment['text']
             current_page = segment['page']

        # If the current element is a heading/title, the previous text must be a complete paragraph.
        if is_structural:
            if stitched_text != segment['text']: # Avoid duplicating if it was the first segment
                final_content.append({'text': stitched_text, 'type': 'paragraph', 'page': current_page})
            final_content.append(segment)
            stitched_text = ""
            continue

        # If the previous segment was a complete sentence, this one starts a new paragraph.
        if i > 0 and initial_segments[i-1]['text'].endswith(('.', '!', '?', ':', '•')):
             final_content.append({'text': stitched_text, 'type': 'paragraph', 'page': current_page})
             stitched_text = segment['text']
             current_page = segment['page']
        # If the text is part of an ongoing paragraph, stitch it.
        elif stitched_text != segment['text']:
            stitched_text += f" {segment['text']}"

    # Add any remaining stitched text at the end of the document
    if stitched_text:
        final_content.append({'text': stitched_text, 'type': 'paragraph', 'page': current_page})
        
    return final_content