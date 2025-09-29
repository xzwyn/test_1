from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

AlignedPair = Dict[str, Any]
EvaluationFinding = Dict[str, Any]

def save_alignment_report(aligned_data: List[AlignedPair], filepath: Path) -> None:
    if not aligned_data:
        print("Warning: No aligned data to save to Excel.")
        return

    report_data = []
    for pair in aligned_data:
        eng_item = pair.get('english')
        ger_item = pair.get('german')
        
        report_data.append({
            "English": eng_item.get('text', '') if eng_item else "--- OMITTED ---",
            "German": ger_item.get('text', '') if ger_item else "--- ADDED ---",
            "Similarity": f"{pair.get('similarity', 0.0):.4f}",
            "Type": (eng_item.get('type') if eng_item else ger_item.get('type', 'N/A')),
            "English Page": (eng_item.get('page') if eng_item else 'N/A'),
            "German Page": (ger_item.get('page') if ger_item else 'N/A')
        })

    df = pd.DataFrame(report_data)
    
    try:
        df.to_excel(filepath, index=False, engine='openpyxl')
    except Exception as e:
        print(f"Error: Could not write alignment report to '{filepath}'. Reason: {e}")

def save_evaluation_report(evaluation_results: List[EvaluationFinding], filepath: Path) -> None:
    if not evaluation_results:
        print("No evaluation findings to save.")
        return

    # Sort results by page number for a more logical report
    evaluation_results.sort(key=lambda x: x.get('page', 0))

    df = pd.DataFrame(evaluation_results)

    desired_columns = [
        "page", "type", "suggestion", "english_text", "german_text", 
        "original_phrase", "translated_phrase"
    ]
    final_columns = [col for col in desired_columns if col in df.columns]
    df = df[final_columns]
    
    try:
        df.to_excel(filepath, index=False, sheet_name='Evaluation_Findings')
    except Exception as e:
        print(f"Error: Could not write evaluation report to '{filepath}'. Reason: {e}")