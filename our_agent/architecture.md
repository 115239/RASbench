# Architecture

The archived pipeline connects a task instruction and sensor observations to navigation and manipulation interfaces. Manipulation predictions pass through the task's action conversion and execution interface. The safety-agent variant adds observation/action checks and bounded intervention at the integration layer.

The paper describes GR00T with the safety agent enabled, GR00T with it disabled, and a π0.5 configuration. Shared components and module-specific code are available in `safety_agent/` and `inference/baseline_runner/`; runtime parameters and the π0.5 transfer mapping are in `configs/`.

The released source is an archived candidate. Exact final-paper checkpoint bindings and the full historical run matrix are unavailable. The additional offline adapters use a separately recorded [training protocol](experiments/market_offline/new_experiment/campaign_v1/PROTOCOL.json).
