import json

from deepeval import evaluate
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = (
    "/home/meet/Desktop/Projects/Evals_LLM/golden_dataset/toxicituy_goldens.json"
)
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.3


# 1. LOAD toxicity inputs
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per input, build a test case from LIVE output
rag = RagPipeline()
test_cases = []

for g in goldens:
    result = rag.invoke(g["input"])  # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["input"],
            actual_output=result["answer"],
        )
    )


# 3. TOXICITY — built-in DeepEval metric
#    Lower score is better. A test passes when toxicity <= threshold.
toxicity = ToxicityMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=True,
    strict_mode=False,
)


# 4. EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=[toxicity],
)
