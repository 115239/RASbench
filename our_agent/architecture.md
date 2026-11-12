# Architecture

The pipeline connects a task instruction and sensor observations to navigation and manipulation interfaces. Manipulation predictions pass through the task's action conversion and execution interface. The safety-agent variant adds observation/action checks and bounded intervention at the integration layer.

The paper describes GR00T with the safety agent enabled, GR00T with it disabled, and a π0.5 configuration. Shared components and module-specific code are available in `safety_agent/` and `inference/baseline_runner/`; runtime parameters and the π0.5 transfer mapping are in `configs/`.

The Q17–Q24 adapters use the [training protocol](experiments/market_offline/new_experiment/campaign_v1/PROTOCOL.json) described in the offline comparison.
