# Evaluation

The `paper200/` package contains the archived collection and evidence-validation implementation. Pairing checks are implemented in [paired_initial.py](paper200/paired_initial.py) and [formal_reference_pairing.py](paper200/formal_reference_pairing.py).

A valid comparison requires matching task/system/seed identities, compatible runtime and rule versions, verified initial state, and the evidence required by the metric. Collection completion and statistical eligibility are distinct statuses.

## Required result records

Per-system and per-interface reporting requires registered interventions, activated interventions, valid pairs, clean-safe pairs, excluded attempts with reasons, and the numerator/denominator for each metric. A shared clean run remains linked to every intervention that reuses it. Repeated observations do not increase the number of independent episodes.

The historical Table 4 records are described in [result availability](../results/README.md). The new Q17–Q24 offline action experiment has a separate [reproduction entrypoint](../../our_agent/experiments/market_offline/README.md).

## External runtime paths

Historical collection entrypoints use `PAPER200_PROJECT_ROOT` for the external simulator workspace. Set `RASBENCH_OPENPI_PYTHON`, `RASBENCH_OPENPI_SOURCE`, and `RASBENCH_PI_TOKENIZER` to the installed OpenPI interpreter, source tree, and tokenizer when using the corresponding model/provenance commands. These variables replace machine-specific source paths.
