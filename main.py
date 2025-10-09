import argparse
import time
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import config
from src.processing.json_parser import process_document_json
from src.alignment.semantic_aligner import align_content, ContentItem  # Import ContentItem from here
from src.reporting.markdown_writer import save_to_markdown
from src.reporting.excel_writer import save_alignment_report, save_evaluation_report, save_calculation_report
from src.evaluation.pipeline import run_evaluation_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Aligns and optionally evaluates content from two Azure Document Intelligence JSON files."
    )
    parser.add_argument("english_json", type=str, help="Path to the English JSON file.")
    parser.add_argument("german_json", type=str, help="Path to the German JSON file.")
    parser.add_argument(
        "-o", "--output", type=str, help="Path for the output alignment Excel file.",
        default=None
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run the AI evaluation pipeline after alignment."
    )
    parser.add_argument(
        "--debug-report", action="store_true",
        help="Generate a detailed Excel report showing the score calculations for debugging."
    )
    # New arguments for alignment options
    parser.add_argument(
        "--algorithm", type=str, choices=["mutual", "hungarian"], default="mutual",
        help="Alignment algorithm to use: mutual best match or Hungarian algorithm."
    )
    parser.add_argument(
        "--context-window", type=int, default=0,
        help="Size of context window (0 for no context, 1+ for context-aware embeddings)."
    )
    parser.add_argument(
        "--compare-methods", action="store_true",
        help="Compare different alignment methods and save results."
    )
    args = parser.parse_args()

    # --- 1. Setup Paths ---
    eng_path = Path(args.english_json)
    ger_path = Path(args.german_json)

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_alignment_path = Path(args.output)
    else:
        output_alignment_path = output_dir / f"alignment_{eng_path.stem}_{timestamp}.xlsx"

    output_md_eng_path = output_dir / f"{eng_path.stem}_processed.md"
    output_md_ger_path = output_dir / f"{ger_path.stem}_processed.md"

    # Path for the debug report
    if args.debug_report:
        output_debug_path = output_dir / f"debug_calculations_{eng_path.stem}_{timestamp}.xlsx"
        print(f"Debug Report will be saved to: {output_debug_path}\n")
    else:
        output_debug_path = None

    print("--- Document Alignment Pipeline Started ---")
    print(f"English Source: {eng_path}")
    print(f"German Source:  {ger_path}")
    print(f"Output Alignment Report:  {output_alignment_path}")
    print(f"Alignment Algorithm: {args.algorithm}")
    print(f"Context Window Size: {args.context_window}\n")

    try:
        print("Step 1/5: Processing JSON files...")
        english_content = process_document_json(eng_path)
        german_content = process_document_json(ger_path)
        print(f"-> Extracted {len(english_content)} English segments and {len(german_content)} German segments.\n")
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}")
        return
    except Exception as e:
        print(f"An error occurred during JSON processing: {e}")
        return

    print("Step 2/5: Creating verification Markdown files...")
    save_to_markdown(english_content, output_md_eng_path)
    save_to_markdown(german_content, output_md_ger_path)
    print(f"-> Markdown files saved in '{output_dir.resolve()}'\n")

    # If comparison mode is enabled, run the comparison and exit
    if args.compare_methods:
        print("Running comparison of alignment methods...")
        compare_alignment_methods(
            english_content,
            german_content,
            output_dir,
            eng_path.stem,
            args.evaluate
        )
        return

    print("Step 3/5: Performing semantic alignment...")
    aligned_pairs = align_content(
        english_content,
        german_content,
        algorithm=args.algorithm,
        context_window=args.context_window,
        generate_debug_report=args.debug_report,
        debug_report_path=output_debug_path
    )
    print(f"-> Alignment complete. Found {len(aligned_pairs)} aligned pairs.\n")

    print("Step 4/5: Writing alignment report to Excel...")
    save_alignment_report(aligned_pairs, output_alignment_path)
    print(f"-> Alignment report saved to: {output_alignment_path.resolve()}\n")

    if args.evaluate:
        print("Step 5/5: Running AI evaluation pipeline...")
        try:
            evaluation_results = list(run_evaluation_pipeline(aligned_pairs))

            if not evaluation_results:
                print("-> Evaluation complete. No significant errors were found.")
            else:
                print(f"-> Evaluation complete. Found {len(evaluation_results)} potential errors.")
                output_eval_path = output_dir / f"evaluation_report_{eng_path.stem}_{timestamp}.xlsx"
                save_evaluation_report(evaluation_results, output_eval_path)
                print(f"-> Evaluation report saved to: {output_eval_path.resolve()}")

        except RuntimeError as e:
            print(f"\nERROR: Could not run evaluation. {e}")
            print("Please ensure your AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set in the .env file.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during evaluation: {e}")

    print("\n--- Pipeline Finished Successfully ---")

def compare_alignment_methods(
    english_content: List[ContentItem],
    german_content: List[ContentItem],
    output_dir: Path,
    base_filename: str,
    run_evaluation: bool = False
):
    """
    Compare different alignment methods and save results.

    Args:
        english_content: List of English content items
        german_content: List of German content items
        output_dir: Directory to save output files
        base_filename: Base name for output files
        run_evaluation: Whether to run evaluation on each alignment
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    methods = [
        {"name": "mutual", "context": 0, "desc": "Mutual Best Match"},
        {"name": "hungarian", "context": 0, "desc": "Hungarian Algorithm"},
        {"name": "mutual", "context": 1, "desc": "Mutual Best Match with Context"},
        {"name": "hungarian", "context": 1, "desc": "Hungarian Algorithm with Context"}
    ]

    results = []

    for method in methods:
        print(f"\n--- Testing: {method['desc']} ---")

        # Perform alignment using the current method
        aligned_pairs = align_content(
            english_content,
            german_content,
            algorithm=method["name"],
            context_window=method["context"]
        )

        # Save alignment report
        output_path = output_dir / f"alignment_{base_filename}_{method['name']}_ctx{method['context']}_{timestamp}.xlsx"
        save_alignment_report(aligned_pairs, output_path)
        print(f"-> Alignment report saved to: {output_path}")

        # Calculate statistics
        total_pairs = len(aligned_pairs)
        matched_pairs = sum(1 for pair in aligned_pairs if pair.get('english') and pair.get('german'))
        unmatched_eng = sum(1 for pair in aligned_pairs if pair.get('english') and not pair.get('german'))
        unmatched_ger = sum(1 for pair in aligned_pairs if not pair.get('english') and pair.get('german'))

        # Run evaluation if requested
        eval_errors = 0
        if run_evaluation:
            try:
                print(f"Running evaluation for {method['desc']}...")
                evaluation_results = list(run_evaluation_pipeline(aligned_pairs))
                eval_errors = len(evaluation_results)

                if evaluation_results:
                    eval_path = output_dir / f"eval_{base_filename}_{method['name']}_ctx{method['context']}_{timestamp}.xlsx"
                    save_evaluation_report(evaluation_results, eval_path)
                    print(f"-> Evaluation report saved to: {eval_path}")
            except Exception as e:
                print(f"Evaluation error: {e}")

        # Store results for comparison
        results.append({
            "Method": method['desc'],
            "Total Pairs": total_pairs,
            "Matched Pairs": matched_pairs,
            "Unmatched English": unmatched_eng,
            "Unmatched German": unmatched_ger,
            "Match Rate": f"{matched_pairs/(matched_pairs+unmatched_eng+unmatched_ger):.2%}",
            "Evaluation Errors": eval_errors if run_evaluation else "N/A"
        })

        print(f"-> {method['desc']}: {matched_pairs} matched pairs, {unmatched_eng} unmatched English, {unmatched_ger} unmatched German")

    # Save comparison report
    comparison_df = pd.DataFrame(results)
    comparison_path = output_dir / f"comparison_{base_filename}_{timestamp}.xlsx"
    comparison_df.to_excel(comparison_path, index=False)
    print(f"\nComparison report saved to: {comparison_path}")

    return comparison_df

if __name__ == "__main__":
    main()
