# Murphy-Health — Longevity Benchmark CLI Dev Log

Iterative refinement benchmark for Longevity-LLM (Track 01 · Insilico Medicine Hackathon).
A CLI that evaluates any model on aging-biology tasks from LongeBench or custom interval tasks.

---

## 2026-05-23 — Initial CLI Design

### Context
The project idea (see `idea.md`) proposes an iterative refinement benchmark on top of
LongeBench: instead of one-shot Q&A, the model submits intervals [min, max] for numerical
tasks and receives yes/no feedback each round. The benchmark tests calibration and convergence,
not just final-answer accuracy.

Before writing any code, we evaluated the idea against Track 01's four scoring criteria:
- **Utility**: strong — tests a genuine failure mode (right answer, wrong reasoning) that
  one-shot metrics miss entirely.
- **Diversity**: strong for numerical tasks (AnAge lifespan, DrugAge extension %); needs care
  to avoid mixing binary/MCQ formats under the same estimathon scoring formula.
- **Retrieval resistance**: strong — interval format makes memorisation hard to exploit even
  if the ground truth value was in pretraining data.
- **Statistical rigor**: clean — estimathon score is mathematically well-defined; needs a
  baseline (e.g. random interval) reported for interpretability.

**Risk flagged**: NHANES and GEO are already in LongeBench — using them as sources would
hurt the Retrieval Resistance score. Decided to target AnAge and DrugAge instead.

### CLI design decisions

**Framework**: Typer + Rich, mirroring SPEQTRO CLI architecture.
- Config persisted at `~/.longevity/config.json`, env vars override.
- JSONL result files (one record per task), same pattern as SPEQTRO trajectories.
- `ThreadPoolExecutor(max_workers=8)` for concurrency — hard cap matches shared endpoint policy.

**Model abstraction**: all providers go through OpenAI SDK by setting `base_url`.
- Open-source models (HuggingFace): `https://api-inference.huggingface.co/models/{id}/v1`
- Custom endpoints (L-LLM, vLLM): user-provided URL
- OpenAI: standard SDK defaults
- Anthropic: separate `anthropic` SDK, mapped to same return shape

**Two eval modes**:
- `one-shot`: standard LongeBench evaluation — send prompt, score final answer
- `iterative`: full feedback loop — interval submission, yes/no signal, convergence tracking

### Files created this session
- `devlog.md` (this file)
- `requirements.txt`
- `benchmark/config.py`
- `benchmark/client.py`
- `benchmark/loader.py`
- `benchmark/runner.py`
- `benchmark/results.py`
- `cli.py`

### Next steps
- Build dataset (Area 1): pull AnAge + DrugAge, design interval task prompts, write tasks.jsonl
- Test CLI against L-LLM endpoint with `--limit 5` dry-run
- Implement trace scorer (Area 3) once first results.jsonl is collected

---
