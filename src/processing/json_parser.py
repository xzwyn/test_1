import json
from pathlib import Path
from typing import List, Dict, Any
import config

ContentItem = Dict[str, Any]

def process_document_json(filepath: Path) -> List[ContentItem]:
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
    
    page_lookup = {}
    for page in pages:
        for span in page['spans']:
            for i in range(span['offset'], span['offset'] + span['length']):
                page_lookup[i] = page['pageNumber']

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
    final_content: List[ContentItem] = []
    stitched_text = ""
    current_page = 0

    for i, segment in enumerate(initial_segments):
        is_structural = segment['type'] in config.STRUCTURAL_ROLES
        text_ends_sentence = segment['text'].endswith(('.', '!', '?', ':', '•'))

        if stitched_text == "": 
             stitched_text = segment['text']
             current_page = segment['page']

        if is_structural:
            if stitched_text != segment['text']: 
                final_content.append({'text': stitched_text, 'type': 'paragraph', 'page': current_page})
            final_content.append(segment)
            stitched_text = ""
            continue

        if i > 0 and initial_segments[i-1]['text'].endswith(('.', '!', '?', ':', '•')):
             final_content.append({'text': stitched_text, 'type': 'paragraph', 'page': current_page})
             stitched_text = segment['text']
             current_page = segment['page']

        elif stitched_text != segment['text']:
            stitched_text += f" {segment['text']}"

    if stitched_text:
        final_content.append({'text': stitched_text, 'type': 'paragraph', 'page': current_page})
        
    return final_content