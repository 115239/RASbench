# RASBench

RASBench evaluates task performance and physical safety under controlled interventions in embodied-agent interfaces.

| Directory | Contents |
| --- | --- |
| [benchmark](benchmark/README.md) | 200 task descriptions, safety monitoring, intervention code, runner, judge, and evaluation protocol |
| [our_agent](our_agent/README.md) | Agent implementation, configuration, competition container, and model comparison results |
| [scenes](scenes/README.md) | Scene previews and image/video archives |

## Reproduce the offline comparison

Python 3.11 or later and NumPy are sufficient to recompute the Q17–Q24 result tables from the included predictions:

```bash
python -m pip install numpy
python our_agent/experiments/market_offline/scripts/aggregate_new_experiment.py
```

The experiment uses 80 held-out episodes and three training seeds per model. See the [protocol and results](our_agent/experiments/market_offline/README.md).

## Benchmark version

The published runner and judge are the archived `runner_v0925` candidate implementation. The [version record](benchmark/provenance/source_status.json) identifies its status. Episode-level inputs needed to reconstruct the paper's Table 4 are currently unavailable in this release; see [result availability](benchmark/results/README.md).

Code and asset licensing is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
