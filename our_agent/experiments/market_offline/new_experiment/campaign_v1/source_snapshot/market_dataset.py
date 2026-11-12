"""Portable synchronized episodes and the matching GR00T IterableDataset adapter."""

from pathlib import Path
from io import BytesIO
from collections import OrderedDict, defaultdict
import json, zipfile
import numpy as np

CAMERAS = ("image", "rear_image", "wrist_image")
STATE_GROUPS = {
    "base_xyz": (0, 3),
    "base_rpy": (3, 6),
    "base_velocity": (6, 12),
    "piper_joint": (12, 18),
    "tcp_xyz": (18, 21),
    "tcp_rpy": (21, 24),
    "gripper_state": (24, 25),
}
ACTION_GROUPS = {
    "base_command": (0, 3),
    "tcp_delta_command": (3, 9),
    "gripper_command": (9, 10),
}
GROUPS = {
    "align": ("LOCAL_ALIGN", "SETTLE", "STABILIZE"),
    "grasp": (
        "ARM_APPROACH",
        "UNSTOW",
        "PREGRASP",
        "CONTACT_CLOSE",
        "GRASP",
        "LIFT_VERIFY",
        "HOLD",
    ),
    "retreat": (
        "RETRACT_LOADED",
        "CARRY_VERIFY",
        "BASE_RETREAT",
        "RETURN_READY_HOLD",
        "TURN_LOADED",
        "LOADED_TURN",
        "AISLE_CENTER",
        "EXIT_SHELF",
        "FOLD_ARM",
    ),
}


def action_end(a, start, horizon=15, mask_key="train_mask"):
    end = start
    for i in range(start, min(start + horizon, len(a["action"]))):
        if not a[mask_key][i] or any(
            a[k][i] != a[k][start]
            for k in ["phase", "phase_epoch", "segment_id", "control_source", "prompt"]
        ):
            break
        end = i + 1
    return end


class EpisodeReader:
    def __init__(self, root=None, profile="train_native"):
        self.root = Path(root or Path(__file__).resolve().parents[1] / "data/grasp_755")
        self.delivery = self.root.parent
        self.entries = [
            json.loads(x)
            for x in (self.root / "meta" / f"{profile}.jsonl").read_text().splitlines()
        ]
        self.cache = OrderedDict()
        self.zips = OrderedDict()
        self.profile = profile

    def arrays(self, e):
        if e not in self.cache:
            with np.load(
                self.root / self.entries[e]["arrays"], allow_pickle=False
            ) as z:
                self.cache[e] = {k: z[k] for k in z.files}
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
        self.cache.move_to_end(e)
        return self.cache[e]

    def read(self, e, step, horizon=15):
        entry = self.entries[e]
        a = self.arrays(e)
        mask = "train_mask"
        end = action_end(a, step, horizon, mask)
        assert end > step, "selected frame is not eligible supervision"
        if entry["format"] == "native_request_zip":
            if e not in self.zips:
                self.zips[e] = zipfile.ZipFile(self.root / entry["observations"])
                if len(self.zips) > 8:
                    self.zips.popitem(last=False)[1].close()
            self.zips.move_to_end(e)
            with np.load(
                BytesIO(self.zips[e].read(f"frames/{step:06d}.npz")), allow_pickle=False
            ) as z:
                images = {c: z[c] for c in CAMERAS}
                assert np.array_equal(z["state"], a["state"][step])
                assert str(z["prompt"]) == str(a["prompt"][step])
        else:
            raise ValueError("only verified three-view native requests are supported")
        return {
            "images": images,
            "state": a["state"][step],
            "actions": a["action"][step:end],
            "prompt": str(a["prompt"][step]),
            "phase": str(a["phase"][step]),
            "phase_epoch": int(a["phase_epoch"][step]),
            "timestamp": float(a["timestamp"][step]),
            "episode_id": entry["episode_id"],
            "frame_index": step,
        }

    def close(self):
        for z in self.zips.values():
            z.close()
        self.zips.clear()


class SamplingPlan:
    def __init__(self, reader, mix=(1, 1, 1), seed=20260909):
        self.seed = seed
        self.mix = np.asarray(mix, float)
        assert self.mix.shape == (3,) and np.all(self.mix > 0)
        buckets = defaultdict(list)
        for e, entry in enumerate(reader.entries):
            a = reader.arrays(e)
            mask = a["train_mask"]
            for phase in sorted(set(a["phase"][mask].tolist())):
                group = next(
                    (g for g, ps in GROUPS.items() if phase in ps),
                    "retreat"
                    if any(
                        k in phase
                        for k in [
                            "RETREAT",
                            "CARRY",
                            "RETRACT",
                            "TURN",
                            "HOLD_READY",
                            "FOLD",
                            "EXIT",
                        ]
                    )
                    else "grasp",
                )
                steps = np.flatnonzero(mask & (a["phase"] == phase))
                buckets[(entry["question"], group, phase)].append((e, steps))
        self.buckets = dict(buckets)
        self.questions = sorted({k[0] for k in buckets})
        assert self.questions, "no usable frames"

    def choice(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        q = self.questions[int(rng.integers(len(self.questions)))]
        gs = [g for g in GROUPS if any(k[:2] == (q, g) for k in self.buckets)]
        w = np.asarray([self.mix[list(GROUPS).index(g)] for g in gs])
        g = gs[int(rng.choice(len(gs), p=w / w.sum()))]
        phases = sorted(k[2] for k in self.buckets if k[:2] == (q, g))
        phase = phases[int(rng.integers(len(phases)))]
        eps = self.buckets[q, g, phase]
        e, frames = eps[int(rng.integers(len(eps)))]
        return e, int(frames[int(rng.integers(len(frames)))])


def register_modality():
    from gr00t.configs.data.embodiment_configs import register_modality_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import (
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
        ModalityConfig,
    )

    cfg = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(CAMERAS)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_GROUPS)),
        "action": ModalityConfig(
            delta_indices=list(range(15)),
            modality_keys=list(ACTION_GROUPS),
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for _ in ACTION_GROUPS
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }
    register_modality_config(cfg, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
    return cfg


def configure_processor(processor, stats):
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    grouped = {
        "new_embodiment": {
            m: {
                g: {k: v[lo:hi] for k, v in stats[m].items() if k != "constant_mask"}
                for g, (lo, hi) in groups.items()
            }
            for m, groups in [("state", STATE_GROUPS), ("action", ACTION_GROUPS)]
        }
    }
    kwargs = {
        "use_percentiles": False,
        "use_relative_action": False,
        "apply_sincos_state_encoding": False,
    }
    for k, v in kwargs.items():
        setattr(processor, k, v)
    processor.state_action_processor = StateActionProcessor(
        modality_configs=processor.modality_configs,
        statistics=grouped,
        clip_outliers=processor.clip_outliers,
        **kwargs,
    )
    processor.set_statistics(grouped, override=True)
    # Match the established constant-channel normalization convention in this processor instance only.
    sap = processor.state_action_processor
    for name in ["apply_action", "unapply_action"]:
        original = getattr(sap, name)

        def wrapped(action, embodiment_tag, state=None, _name=name, _original=original):
            out = _original(action, embodiment_tag, state)
            if embodiment_tag == "new_embodiment":
                for g, (lo, hi) in ACTION_GROUPS.items():
                    mask = np.asarray(stats["action"]["constant_mask"][lo:hi], bool)
                    if mask.any():
                        out[g][..., mask] = (
                            0
                            if _name == "apply_action"
                            else np.asarray(stats["action"]["min"][lo:hi])[mask]
                        )
            return out

        setattr(sap, name, wrapped)
    return grouped


def make_dataset(root, processor, profile="train_native", mix=(1, 1, 1), seed=20260909):
    import torch
    from torch.utils.data import IterableDataset
    from gr00t.data.types import VLAStepData
    from gr00t.data.embodiment_tags import EmbodimentTag

    class Dataset(IterableDataset):
        def __init__(self):
            self.reader = EpisodeReader(root, profile)
            self.plan = SamplingPlan(self.reader, mix, seed)
            self.start_sample = 0
            self.stats = json.loads(
                (self.reader.root / "meta" / f"{profile}_statistics.json").read_text()
            )
            self.global_stats = configure_processor(processor, self.stats)

        def sample(self, e, step):
            r = self.reader.read(e, step)
            data = VLAStepData(
                images={k: [v] for k, v in r["images"].items()},
                states={
                    k: r["state"][None, lo:hi] for k, (lo, hi) in STATE_GROUPS.items()
                },
                actions={
                    k: r["actions"][:, lo:hi] for k, (lo, hi) in ACTION_GROUPS.items()
                },
                text=r["prompt"],
                embodiment=EmbodimentTag.NEW_EMBODIMENT,
            )
            sample = processor([{"type": "episode_step", "content": data}])
            assert not torch.count_nonzero(
                sample["action_mask"][len(r["actions"]) :]
            ), "padding entered loss"
            return sample

        def __iter__(self):
            assert torch.utils.data.get_worker_info() is None, (
                "this deterministic stream uses workers=0"
            )
            i = self.start_sample
            while True:
                yield self.sample(*self.plan.choice(i))
                i += 1

        def get_dataset_statistics(self):
            return self.global_stats

        def get_initial_actions(self):
            return {}

        def set_epoch(self, epoch):
            pass

        def reset_seed(self, value):
            assert value == seed

    return Dataset()
