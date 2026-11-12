# Runner source

The archived runner contains `robolab_runner/` (CLI and recording protocol), `m20_runtime/` (simulation, safety monitoring, and intervention integration), and vendored protocol adapters.

Runtime dependencies are declared in [pyproject.toml](pyproject.toml). Full execution requires the matching Isaac Sim environment, robot and scene assets, executable task contracts, and a policy endpoint. These dependencies are separate from the included offline-result reproduction command.

The source is the `runner_v0925` candidate. Its physical-validation directory was absent from the supplied package. See [source status](../provenance/source_status.json).

The source test suite is in [tests](tests); simulator-dependent tests require the original runtime dependencies.
