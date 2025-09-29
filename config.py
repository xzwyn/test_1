MODEL_NAME: str = 'distiluse-base-multilingual-cased-v2'

SIMILARITY_THRESHOLD: float = 0.70

W_SEMANTIC: float = 0.7
W_TYPE: float = 0.2
W_PROXIMITY: float = 0.1

TYPE_MATCH_BONUS: float = 1.0
TYPE_MISMATCH_PENALTY: float = -1.0

AZURE_CHAT_DEPLOYMENT: str = "gpt-4o"

IGNORED_ROLES = {"pageHeader", "pageFooter", "pageNumber"}
STRUCTURAL_ROLES = {'title', 'sectionHeading', 'subheading'}

INPUT_DIR: str = "input"
OUTPUT_DIR: str = "output"