# Q17–Q24 offline action comparison

## Protocol

640 teacher trajectories are split by episode into 480 training, 80 validation, and 80 test trajectories (60/10/10 per question). Three seeds per model use 2,000 updates, effective batch size 4, learning rate 1e-5, 100 warmup steps, and AdamW. Each model receives a 25D state, three RGB views, the task/phase instruction, and a 15×10 supervised action target.

The parameter-efficient adaptation recipes differ: π0.5 trains 52,153,376 parameters, and GR00T trains 34,460,032. Adapter locations, native image/state preprocessing, internal horizons (15 versus 40), and inference steps (10 versus 4) differ. This is a fixed short-budget comparison.

## Results

The primary test evaluates four fixed observations from each of 80 held-out episodes. Errors are averaged within episode, then across episodes and three seeds. TCP values measure predicted action-increment error, in physical units.

| First-action MAE | π0.5 | GR00T |
| --- | ---: | ---: |
| Base forward velocity (m/s) | 0.0107476 | 0.0429639 |
| Base yaw velocity (rad/s) | 0.0114590 | 0.0780608 |
| TCP translation increment (m) | 0.000601350 | 0.009760716 |
| TCP rotation increment (rad) | 0.001669677 | 0.029147753 |
| Gripper closure fraction | 0.046941537 | 0.281587481 |

π0.5 has lower mean error on all five nonconstant metrics. The lateral-velocity target is constant. Simple action baselines are retained in the prediction package; the phase-mean baseline outperforms the adapted models on several motion metrics. The evaluation measures conditional action prediction; closed-loop task success and autonomous phase selection were not evaluated.

The phase-coverage evaluation contains 802 observations from the same 80 test episodes. These observations do not constitute additional independent episodes. Shared-GPU latency is diagnostic and is not used to claim a deployment-cost advantage.

## Recompute

From the repository root:

```bash
python -m pip install numpy
python our_agent/experiments/market_offline/scripts/aggregate_new_experiment.py
python our_agent/experiments/market_offline/scripts/aggregate_phase_coverage.py
python our_agent/experiments/market_offline/scripts/analyze_action_modes.py
```

[Main results](analysis/new_offline/matched_test_summary.csv), [per-seed results](analysis/new_offline/per_seed.csv), and [per-question results](analysis/new_offline/per_question.csv) are computed from the included prediction NPZ/JSONL files. Confidence intervals use a paired bootstrap across seeds and question-stratified episodes; with three training seeds they are descriptive intervals.

## Training materials

[Protocol](new_experiment/campaign_v1/PROTOCOL.json), [implementation details](new_experiment/campaign_v1/IMPLEMENTATION_DETAILS.json), and [training source](new_experiment/campaign_v1/source_snapshot) describe the executed adaptation. Split indexes retain episode identities and source hashes. Re-training additionally requires the teacher arrays/images, foundation checkpoints, and the original framework dependencies; these are not needed for the statistical recomputation above.

## Adapter download

Six adapters are hosted in GitHub Releases. From the repository root:

```bash
python download.py adapters --output model_adapters
unzip model_adapters/RASBench-market-adapters.zip -d model_adapters
```

[SHA-256](ADAPTER_SHA256SUMS.txt) · [File hashes](ADAPTER_FILES.json)

The adapters require their matching foundation models and the included custom loaders. Optimizer state is not included.
