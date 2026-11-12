"""One public WebSocket endpoint; distinct, bound GR00T navigation/manipulation ABIs."""
import argparse
import asyncio
import base64
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from common import ROOT, WS, VLN, GRASP_CODE, MODEL_PYTHON, read, write, binding

for p in [WS / "code/pylibs", WS / "code", GRASP_CODE, WS / "code/Isaac-GR00T", VLN, VLN / "nav2_outbound"]:
    sys.path.insert(0, str(p))

import numpy as np
from openpi_client import msgpack_numpy
import websockets.asyncio.server as ws

def isolate_navigation_processor(model):
    # G2's constant-channel rule patches StateActionProcessor globally. Capture
    # the original methods on each already-loaded navigation processor so that
    # the grasp-only base_command/eef_delta keys never leak into goal decoding.
    processor = model.policy.processor.state_action_processor
    processor.apply_action = processor.apply_action
    processor.unapply_action = processor.unapply_action

def post_pick_scores(model, images, instruction, carrying):
    """The old worker required loaded=True; this mission also allows empty return.

    Keep the checkpoint's actual six-channel encoding (-1/+1), normalization,
    seeds and decoder. Only remove the old task-level loaded-only precondition.
    """
    import torch
    from return_delivery.inference import action_with_audit
    if type(carrying) is not bool:
        raise ValueError("CARRYING_MUST_BE_BOOL")
    n = len(images)
    observation = {"video": {"front": np.asarray(images, np.uint8)[:, None]},
                   "state": {"phase": np.tile(np.asarray([-1., -1., -1., -1., 1.], np.float32)[None, None], (n, 1, 1)),
                             "carrying": np.full((n, 1, 1), 1. if carrying else -1., np.float32)},
                   "language": {"annotation.human.task_description": [[instruction] for _ in images]}}
    goals = []
    for seed in model.seeds:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        action, _ = action_with_audit(model.policy, observation)
        score = np.asarray(action["goal_scores"])
        if score.shape != (n, 1, 11) or not np.isfinite(score).all():
            raise ValueError("RETURN_MODEL_OUTPUT_ABI")
        goals.append(score[:, 0])
    return np.asarray(goals)

def formal_grasp_context(actor):
    from contestant_adapter import configure_actor
    configure_actor(actor)

class Router:
    def __init__(self, config, output):
        self.cfg, self.output = config, output
        if 'experiment_seed' in config:
            import random
            import torch
            seed = int(config['experiment_seed'])
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.models = {}
        self.history = deque(maxlen=3)
        self.key = None
        self.calls = 0
        # Load all three before simulation; physics does not wait on checkpoint loading.
        need_navigation=any(s.startswith('NAV_') for s in config['task']['stages'])
        need_manipulation=any(s in ('PICK','PLACE','PUSH','BUTTON','OPEN','HANDOFF') for s in config['task']['stages'])
        if need_navigation and config.get("nav_model"):
            from all_scences.model import Selector, register_modality
            register_modality()
            self.models["unified"] = Selector(Path(config["nav_model"]))
        elif need_navigation:
            from market_vln_nav2.groot_vln_worker import GrootSelector
            from return_delivery.worker import ReturnSelector
            self.models["outbound"] = GrootSelector(Path(config["nav_out"]), WS / "external_resources/src/Isaac-GR00T",
                                                      "cuda:0", [20260921, 20260922, 20260923], 4)
            self.models["return"] = ReturnSelector(Path(config["nav_return"]), calibration_path=Path(config["nav_return"]) / "CALIBRATION.json")
        for model in self.models.values():
            isolate_navigation_processor(model)
        if need_manipulation:
            from m20_piper.local_align.server import GrootActor
            self.models["grasp"] = GrootActor(Path(config["grasp_model"]))
            formal_grasp_context(self.models["grasp"])
        self.metadata = {"state_dim": 25, "action_dim": 10, "schema": "continuous-groot-router/v1",
                         "models": config["model_bindings"], "navigation_output": "semantic_goal_only",
                         "grasp_context": "FORMAL_PUBLIC_INSTRUCTION_AND_PHASE_CLOCK",
                         "adjusted_scene_reference_context_enabled": False,
                         "original_instruction_preserved": True,"loaded_networks":sorted(self.models)}

    def validate_abis(self):
        output = self.output
        self.output = output / "abi_only"
        camera = np.zeros((224, 224, 3), np.uint8)
        state = np.zeros(25, np.float32); state[2] = .59
        obs = {"prompt": self.cfg["task"]["instruction"], "observation/state": state,
               "observation/m20_camera_0_rgb": camera, "observation/m20_camera_1_rgb": camera,
               "observation/piper_wrist_rgb": camera}
        checks = []
        for epoch, phase in enumerate(["NAV_OUTBOUND", "NAV_RETURN", "PICK", "PLACE"]):
            for step in range(3 if phase.startswith("NAV_") else 1):
                obs["mission"] = dict(phase=phase, epoch=epoch, scene=self.cfg["task"]["scene"], carrying=False,
                                      time_s=step / self.cfg.get("action_hz", 10), phase_step=0, model_phase="ARM_APPROACH")
                response = self.infer(obs)
                if np.asarray(response["actions"]).shape[-1] != 10 or not np.isfinite(response["actions"]).all():
                    raise ValueError("MODEL_ABI_SMOKE_FAILED:" + phase)
            checks.append({"phase": phase, "action_shape": list(np.asarray(response["actions"]).shape)})
        write(output / "ABI_SMOKE.json", {"synthetic_observation": True, "physical_acceptance": False, "checks": checks})
        self.output = output; self.history.clear(); self.key = None; self.calls = 0

    def infer(self, obs):
        meta = obs["mission"]
        allowed = {"phase", "epoch", "scene", "carrying", "time_s", "phase_step", "model_phase"}
        if set(meta) != allowed:
            raise ValueError("unexpected mission fields")
        phase, epoch = meta["phase"], meta["epoch"]
        if type(epoch) is not int or epoch < 0 or type(meta["carrying"]) is not bool:
            raise ValueError("invalid phase identity")
        key = (phase, epoch, meta["scene"])
        if key != self.key:
            self.history.clear()
            self.key = key
        instruction = str(obs["prompt"])
        image = np.asarray(obs["observation/m20_camera_0_rgb"], dtype=np.uint8)
        response = {"actions": np.zeros((1, 10), np.float32), "phase": phase, "epoch": epoch,
                    "sample_time_s": meta["time_s"], "models": self.cfg["model_bindings"]}
        if phase.startswith("NAV_"):
            if self.history and meta["time_s"] <= self.history[-1][0]:
                raise ValueError("navigation frame is not newer")
            # Periodic inference may be farther apart than the training frame
            # interval. Rebuild a fresh three-frame window instead of mixing
            # stale views (or views from before an instruction replacement).
            if self.history and (meta["time_s"] - self.history[-1][0] > 1. / self.cfg.get("action_hz", 10) + .01
                                 or instruction != self.history[-1][2]):
                self.history.clear()
            self.history.append((meta["time_s"], image.copy(), instruction))
            if meta['scene'] not in {'factory','kitchen','market'}:
                response['selection'] = {'decision':'ABSTAIN','goal_id':None,
                    'reason':'SCENE_OUTSIDE_FROZEN_NAVIGATION_VOCABULARY','scene':meta['scene']}
            elif len(self.history) < 3:
                response["selection"] = {"decision": "WAIT_FOR_FRESH_FRAMES"}
            elif meta["time_s"] - self.history[0][0] > 2. / self.cfg.get("action_hz", 10) + .01:
                raise ValueError("stale navigation camera window")
            else:
                nav_phase = "SELECT_OUTBOUND" if phase == "NAV_OUTBOUND" else "SELECT_RETURN_DESTINATION"
                if "unified" in self.models:
                    from all_scences.model import decode
                    m = self.models["unified"]
                    seed = self.cfg.get("nav_sampling_seed")
                    sampling = {} if seed is None else {"seeds": tuple(int(v) for v in np.random.default_rng(int(seed)).integers(0,2**31-1,size=3))}
                    goals, routes = m.score([h[1] for h in self.history], instruction, meta["scene"], nav_phase, meta["carrying"], **sampling)
                    response["nav_sampling_seeds"] = sampling.get("seeds", (20260921, 20260922, 20260923))
                    selection = decode(goals, meta["scene"], nav_phase, m.calibration, m.vocabulary)
                elif phase == "NAV_OUTBOUND":
                    from market_vln_nav2.groot_vln_worker import CHECKPOINT_ID
                    frames = [{"encoding": "rgb8", "width": int(a.shape[1]), "height": int(a.shape[0]),
                               "data_base64": base64.b64encode(a.tobytes()).decode(), "receipt_wall_timestamp_s": time.time()}
                              for _, a, _ in self.history]
                    text = " ".join(instruction.strip().split())
                    raw = self.models["outbound"].infer({"schema": "m20-market-vln-worker-request/v1", "phase": "SELECT_OUTBOUND",
                        "carrying": False, "model_checkpoint_id": CHECKPOINT_ID, "frames": frames, "instruction": text,
                        "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "mission_id": "continuous",
                        "phase_epoch": epoch, "request_id": str(self.calls)})
                    ids = [v["goal_id"] for v in raw["decisions"]]
                    ok = len(set(ids)) == 1 and ids[0] != "ABSTAIN"
                    selection = {"decision": "ACCEPT" if ok else "ABSTAIN", "goal_id": ids[0] if ok else None,
                                 "diagnostics": raw["diagnostics"]}
                else:
                    from return_delivery.model import decode_scores
                    from return_delivery.contracts import PHASES
                    m = self.models["return"]
                    goals = post_pick_scores(m, [h[1] for h in self.history], instruction, meta["carrying"])
                    selection = decode_scores(goals, PHASES[4], m.calibration)
                response["selection"] = selection
        else:
            from m20_piper.local_align.control import canonical_state
            public = {"state": canonical_state(obs["observation/state"]), "prompt": instruction,
                      "image": image, "rear_image": np.asarray(obs["observation/m20_camera_1_rgb"], dtype=np.uint8),
                      "wrist_image": np.asarray(obs["observation/piper_wrist_rgb"], dtype=np.uint8)}
            response.update(self.models["grasp"].infer(public, meta["phase_step"], meta["model_phase"]))
        self.calls += 1
        write(self.output / f"inference_{self.calls:06d}.json", {"request": meta, "instruction": instruction,
              "action_shape": list(np.asarray(response["actions"]).shape), "selection": response.get("selection"),
              "projector_category": getattr(self.models["grasp"], "last_projector_category", None) if not phase.startswith("NAV_") else None,
              "model_prompt_sha256": getattr(self.models["grasp"], "last_model_prompt_sha256", None) if not phase.startswith("NAV_") else None,
              "models": self.cfg["model_bindings"]})
        return response

async def serve(router, port):
    router.validate_abis()
    async def handle(connection):
        await connection.send(msgpack_numpy.packb(router.metadata))
        async for data in connection:
            obs = msgpack_numpy.unpackb(data)
            try:
                result = router.infer(obs)
            except Exception as e:
                write(router.output / "ERROR.json", {"error": str(e), "traceback": traceback.format_exc()})
                await connection.close(code=1011, reason=str(e)[:100])
                return
            await connection.send(msgpack_numpy.packb(result))
    async with ws.serve(handle, "127.0.0.1", port, compression=None, max_size=32 * 1024**2, ping_interval=None):
        write(router.output / "READY.json", router.metadata)
        await asyncio.Future()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    asyncio.run(serve(Router(read(a.config), a.out), a.port))
