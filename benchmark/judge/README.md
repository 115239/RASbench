# Judge source

The `v2/` package contains recorded-episode scoring, rule-package validation, and result construction. Its Python dependencies are declared in [pyproject.toml](v2/pyproject.toml).

Scoring requires a recorded episode and matching task contract, rule package, safety profile, and evidence. Semantic or geometry-dependent components require their configured services or assets. A score from another task/rule version cannot replace the historical paper score.

This copy comes from the same archived candidate as the [runner](../runner/README.md). The source tree contains implementation code; the final paper-run binding is recorded as unavailable in [source_status.json](../provenance/source_status.json).
