"""Central configuration: paths, model ids, API key."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Bare Gemini model ids (e.g. "gemini-3.1-flash-lite"). LangChain uses them as-is; DSPy/litellm prefixes "gemini/".
TASK_MODEL = os.getenv("TASK_MODEL") or "gemini-3.1-flash-lite"
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or TASK_MODEL
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL") or TASK_MODEL

DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "hotpotqa"
EMBED_MODEL = "minishlab/potion-retrieval-32M"
TOP_K = 4

PROMPTS_DIR = ROOT / "prompts"
BASELINE_PROMPTS = PROMPTS_DIR / "baseline.json"
OPTIMIZED_PROMPTS = PROMPTS_DIR / "optimized.json"
RUNS_DIR = ROOT / "runs"
