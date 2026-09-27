# Universal question generation rinse

Applies to **AP**, **ACT**, **Digital SAT**, and other batch-generated MCQ pools.

## Quality goal

- **Pass:** DeepSeek and Gemini each return `A`–`D` and **both equal `db_answer`**.
- **Diagrams:** same rules as text (no auto-fail for `requires_diagram`).
- **Target:** ≥ **90%** pass rate within **3 iterations**.

## Phases (every iteration)

1. **Dual-model QC** (DeepSeek + Gemini, checkpoint file).
2. If ≥ 90% → **final explanations batch** → stop.
3. **Eliminate** (E1–E4) → delete questions, hints, diagram PNGs.
4. **Grok replace batch** → poll xAI → import (LaTeX repair on parse).
5. **Hints:** delete hint docs for questions missing hints → OpenAI hints batch → poll → import (step-by-step QC).
6. Re-score; after iteration 3 or ≥ 90% → **wrong-choice explanations batch** (last).

## LaTeX

- Generation prompts: inline `\( ... \)` (see ACT Math / AP system prompts).
- Import: `repair_latex_escapes` on batch JSON before load.
- Hints/explanations: same delimiter rules in OpenAI batch prompts.

## ACT Math implementation

- Orchestrator: `scripts/act_math_rinse_cycle.py`
- Rules: `pipeline/review_pipeline/rinse_quality.py`
- QC runner: `pipeline/review_pipeline/run_rinse_provider_qc.py`
- State: `pipeline/review_reports/act_math_rinse_state.json`

```bash
# Third iteration (after two prior manual rinse passes)
.venv/bin/python scripts/act_math_rinse_cycle.py run --iteration 3

.venv/bin/python scripts/act_math_rinse_cycle.py poll
```

Other subjects: copy config pattern and swap framework, collection, and prepare/replace scripts.
