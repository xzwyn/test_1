# document_aligner/src/alignment/semantic_aligner.py

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

import config

# Type aliases for clarity
ContentItem = Dict[str, Any]
AlignedPair = Dict[str, Any]

# Global cache for the model to avoid reloading it
_model = None

def _get_model(model_name: str) -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model."""
    global _model
    if _model is None:
        print(f"Loading sentence transformer model '{model_name}'...")
        _model = SentenceTransformer(model_name)
    return _model

def _calculate_type_matrix(eng_content: List[ContentItem], ger_content: List[ContentItem]) -> np.ndarray:
    """Calculates a bonus/penalty matrix based on content type matching."""
    num_eng = len(eng_content)
    num_ger = len(ger_content)
    type_matrix = np.zeros((num_eng, num_ger))

    for i in range(num_eng):
        for j in range(num_ger):
            if eng_content[i]['type'] == ger_content[j]['type']:
                type_matrix[i, j] = config.TYPE_MATCH_BONUS
            else:
                type_matrix[i, j] = config.TYPE_MISMATCH_PENALTY
    return type_matrix

def _calculate_proximity_matrix(num_eng: int, num_ger: int) -> np.ndarray:
    """Calculates a score matrix based on the relative position of items."""
    proximity_matrix = np.zeros((num_eng, num_ger))
    for i in range(num_eng):
        for j in range(num_ger):
            norm_pos_eng = i / num_eng
            norm_pos_ger = j / num_ger
            # Score is 1 for identical relative positions, 0 for opposite ends of the documents.
            proximity_matrix[i, j] = 1.0 - abs(norm_pos_eng - norm_pos_ger)
    return proximity_matrix

def align_content(
    english_content: List[ContentItem],
    german_content: List[ContentItem]
) -> List[AlignedPair]:
    """
    Aligns two lists of content items using a hybrid scoring model.

    The final score for each potential pair is a weighted average of:
    1. Semantic Similarity (what it means)
    2. Type Similarity (how it's structured)
    3. Positional Proximity (where it is)

    Args:
        english_content: A list of structured content items from the English document.
        german_content: A list of structured content items from the German document.

    Returns:
        A list of dictionaries, each representing an aligned pair, an omission, or an addition.
    """
    if not english_content or not german_content:
        return []

    model = _get_model(config.MODEL_NAME)
    num_eng, num_ger = len(english_content), len(german_content)

    # --- 1. Generate Embeddings for Semantic Score ---
    eng_texts = [item['text'] for item in english_content]
    ger_texts = [item['text'] for item in german_content]
    
    print("Generating embeddings...")
    english_embeddings = model.encode(eng_texts, convert_to_numpy=True, show_progress_bar=True)
    german_embeddings = model.encode(ger_texts, convert_to_numpy=True, show_progress_bar=True)
    
    # --- 2. Calculate Individual Score Matrices ---
    print("Calculating score matrices (semantic, type, proximity)...")
    semantic_matrix = cosine_similarity(english_embeddings, german_embeddings)
    type_matrix = _calculate_type_matrix(english_content, german_content)
    proximity_matrix = _calculate_proximity_matrix(num_eng, num_ger)

    # --- 3. Calculate Blended Score Matrix ---
    blended_matrix = (
        (config.W_SEMANTIC * semantic_matrix) +
        (config.W_TYPE * type_matrix) +
        (config.W_PROXIMITY * proximity_matrix)
    )

    # --- 4. Find Best Matches using the Blended Score ---
    print("Finding best matches based on blended scores...")
    aligned_pairs: List[AlignedPair] = []
    used_german_indices = set()
    
    best_ger_matches = np.argmax(blended_matrix, axis=1)
    best_eng_matches = np.argmax(blended_matrix, axis=0)

    for eng_idx, ger_idx in enumerate(best_ger_matches):
        is_mutual_best_match = (best_eng_matches[ger_idx] == eng_idx)
        score = blended_matrix[eng_idx, ger_idx]

        if is_mutual_best_match and score >= config.SIMILARITY_THRESHOLD:
            # We use the raw semantic score for reporting, as it's more interpretable
            semantic_score = semantic_matrix[eng_idx, ger_idx]
            aligned_pairs.append({
                "english": english_content[eng_idx],
                "german": german_content[ger_idx],
                "similarity": semantic_score  # Report the original semantic score
            })
            used_german_indices.add(ger_idx)

    # --- 5. Identify Omissions and Additions ---
    matched_english_indices = {id(pair['english']) for pair in aligned_pairs if pair.get('english')}
    for item in english_content:
        if id(item) not in matched_english_indices:
            aligned_pairs.append({"english": item, "german": None, "similarity": 0.0})

    for idx, item in enumerate(german_content):
        if idx not in used_german_indices:
             aligned_pairs.append({"english": None, "german": item, "similarity": 0.0})

    # Sort the final list for logical order
    aligned_pairs.sort(key=lambda x: x['english']['page'] if x.get('english') else float('inf'))
    
    return aligned_pairs