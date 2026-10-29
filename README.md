# Context enrichment and selective robustness in LLM social agents

This repository contains the materials for the hospital pilot reported in *A Framework for Context Enrichment and Selective Robustness in LLM Social Agents*. The experiment crosses two social worlds, five context conditions, and four meaning-preserving surface forms. It queries `gpt-5.6-sol` and `gpt-5.6-terra` ten times per cell, for 800 fresh calls. The outcome is a conditional first-action-branch estimate of `TRANSFER` probability from returned token log-probabilities.

## Contents

- `tests/fixtures/prompts/hospital/pilot_v4/`: all 40 frozen model-facing prompts and their hashes.
- `src/selective_robustness/` and `scenarios/selective_robustness/conf/`: prompt generator, OpenAI Responses adapter, and native SiliSocS scenario.
- `experiments/studies/hospital_pilot/probability_design.yaml` and `configs/hospital-probability.yaml`: the paper's probability protocol and sampling settings.
- `data/paper_calls.csv`: the 800 call-level probability estimates and censoring bounds.
- `data/paper_results.json`: the published cell-level measures and bootstrap intervals.
- `analysis/reproduce.py` and `analysis/censoring.py`: validate the released data, recompute the measures and censoring ranges, run a fresh 10,000-resample within-cell bootstrap, and generate a figure.

Earlier development pilots and action-frequency experiments are not part of this release.

## Install and verify

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) are required. The package pins upstream [SiliSocS](https://github.com/sandbox-social/silisocs) at commit `ebe652257eafe0eb64e43e90b45aa7bccd732383`; its source is installed as a dependency and is not vendored here.

```bash
uv sync --all-extras
uv run selective-robustness validate
uv run selective-robustness probability-plan
uv run selective-robustness smoke
uv run python analysis/reproduce.py
uv run python analysis/censoring.py
```

The commands above make no model API calls. The smoke command uses a scripted SiliSocS model and writes to ignored `results/`. The analysis writes `results/reproduction/recomputed.json` and a PDF figure. Its point estimates are checked against `data/paper_results.json` to numerical precision. The newly resampled bootstrap endpoints can differ slightly from those stored in `paper_results.json` because the original bootstrap random seed was not retained in the saved summary; the method and 10,000-resample count are the same.

## Rerun the model experiment

The original collection used two independent five-call batches for each model. Prepare and execute a fresh batch for Sol as follows, after setting `OPENAI_API_KEY` in your environment:

```bash
uv run selective-robustness pilot-prepare \
  --measurement-protocol first_action_branch_v1 \
  --model gpt-5.6-sol \
  --temperature 0.7 \
  --reasoning-effort none \
  --max-output-tokens 64 \
  --timeout-seconds 60 \
  --top-logprobs 20 \
  --calls-per-template 5 \
  --maximum-total-calls 200 \
  --randomization-seed 2026082701 \
  --output results/sol-batch-1 \
  --allow-paid \
  --acknowledge-paid-pilot I_ACKNOWLEDGE_SCHEDULED_PAID_PILOT \
  --acknowledge-reviewed-prompt

uv run selective-robustness pilot-run \
  --manifest results/sol-batch-1/run_manifest.json \
  --allow-paid \
  --acknowledge-paid-pilot I_ACKNOWLEDGE_SCHEDULED_PAID_PILOT

uv run selective-robustness probability-summarize \
  --manifest results/sol-batch-1/run_manifest.json \
  --output results/sol-batch-1/summary.json
```

Repeat with `gpt-5.6-terra` and a separate output directory, then run a second five-call batch for each model with new output directories and randomization seeds. The schedule seed orders cells; it does not seed provider responses. The requests omit `top_p`, matching the original protocol. All model calls are paid and create new stochastic observations, so they will not reproduce the released values exactly. The runner writes immutable manifests and append-only call ledgers under ignored `results/`.

The original data and censoring details are described in [data/README.md](data/README.md). The source code is MIT licensed; see [LICENSE](LICENSE).
