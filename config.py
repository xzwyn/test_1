import os
from dotenv import load_dotenv

load_dotenv()

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")

IGNORED_ROLES = {'pageHeader', 'pageFooter', 'pageNumber'}
STRUCTURAL_ROLES = {'title', 'sectionHeading'}

W_SEMANTIC = 0.98  # Weight for semantic similarity (cosine score)
W_TYPE = 0.01      # Weight for matching content types (e.g., table vs. table)
W_PROXIMITY = 0.01 # Weight for relative position in the document

TYPE_MATCH_BONUS = 0.01
TYPE_MISMATCH_PENALTY = -0.01

# The minimum blended score for a pair to be considered a match
SIMILARITY_THRESHOLD = 0.7

INPUT_DIR: str = "input"
OUTPUT_DIR: str = "output"
