import json
from typing import List, Dict, Any
from tqdm import tqdm

# Updated import paths to match the new structure
from src.evaluation.evaluators import evaluate_translation_pair, check_context_mismatch
from src.clients.azure_client import chat

AlignedPair = Dict[str, Any]
EvaluationFinding = Dict[str, Any]

def _agent2_validate_finding(
    eng_text: str,
    ger_text: str,
    error_type: str,
    explanation: str,
    model_name: str | None = None,
):
    """Second-stage reviewer. Confirms only truly fatal errors."""
    prompt = f"""
## ROLE
**Senior Quality Reviewer** – you are the final gatekeeper of EN→DE
...
(The rest of your detailed prompt for Agent 2 is unchanged)
...
## YOUR RESPONSE
Return the JSON object only – no extra text.
"""
    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=model_name,
        )
        j0, j1 = content.find("{"), content.rfind("}") + 1
        verdict_json = json.loads(content[j0:j1])
        is_confirmed = verdict_json.get("verdict", "").lower() == "confirm"
        reasoning = verdict_json.get("reasoning", "")
        return is_confirmed, reasoning, content.strip()
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"  - Agent-2 JSON parse error: {exc}")
        return False, f"System error: {exc}", "{}"
    except Exception as exc:
        print(f"  - Agent-2 unexpected error: {exc}")
        return False, "System error (non-parsing issue)", "{}"

def run_evaluation_pipeline(aligned_pairs: List[AlignedPair]) -> List[EvaluationFinding]:
    """
    Runs the full multi-agent evaluation pipeline on the aligned document pairs.
    """
    findings = []
    
    # Using tqdm for a progress bar
    for pair in tqdm(aligned_pairs, desc="Evaluating Pairs"):
        # --- COMPATIBILITY CHANGE: Unpack pair from dict structure ---
        eng_elem = pair.get('english')
        ger_elem = pair.get('german')

        # Case 1: An English item was omitted in the German document
        if eng_elem and not ger_elem:
            findings.append({
                "type": f"Omission",
                "english_text": eng_elem['text'],
                "german_text": "---",
                "suggestion": "This content from the English document is missing in the German document.",
                "page": eng_elem['page']
            })
            continue
        
        # Case 2: An extra German item was added that doesn't exist in English
        if not eng_elem and ger_elem:
            findings.append({
                "type": f"Addition",
                "english_text": "---",
                "german_text": ger_elem['text'],
                "suggestion": "This content from the German document does not appear to have a source in the English document.",
                "page": ger_elem['page']
            })
            continue

        # Case 3: A matched pair to evaluate
        if eng_elem and ger_elem:
            eng_text = eng_elem['text']
            ger_text = ger_elem['text']

            # NOTE: The original code had table evaluation logic.
            # Our current json_parser does not extract tables. This can be added later.
            
            # Agent 1
            finding = evaluate_translation_pair(eng_text, ger_text)
            error_type = finding.get("error_type", "None")

            if error_type not in ["None", "System Error"]:
                # Agent 2
                is_confirmed, reasoning, _ = _agent2_validate_finding(
                    eng_text, ger_text, error_type, finding.get("explanation")
                )

                if is_confirmed:
                    findings.append({
                        "type": error_type,
                        "english_text": eng_text,
                        "german_text": ger_text,
                        "suggestion": finding.get("suggestion"),
                        "page": eng_elem.get('page'),
                        "original_phrase": finding.get("original_phrase"),
                        "translated_phrase": finding.get("translated_phrase")
                    })
                else: # Agent 2 rejected, run Agent 3
                    context_result = check_context_mismatch(eng_text, ger_text)
                    context_match_verdict = context_result.get('context_match', 'Error')
                    if context_match_verdict.lower() == "no":
                        findings.append({
                            "type": "Context Mismatch",
                            "english_text": eng_text,
                            "german_text": ger_text,
                            "suggestion": context_result.get("explanation"),
                            "page": eng_elem.get('page')
                        })

    return findings