"""资产网格 -> 同一 render product 的 RGB/实例分割 -> 模型输入掩码。"""
from pathlib import Path
import numpy as np
from PIL import Image
from attack_tools.bboard_visual import pack_mask

class BBoardCamera:
    def __init__(self, collector, config, evidence):
        from pxr import Usd, UsdGeom, Semantics
        import omni.replicator.core as rep
        self.c = collector
        self.config = config
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.mesh_paths = []
        stage = collector.final_stage
        # 仅在已有 session layer 中添加语义标记，不写磁盘 USD、不改变材质或碰撞。
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for path in config["target_prims"]:
                target = stage.GetPrimAtPath(path)
                if not target:
                    raise RuntimeError("BBOARD_VISUAL_TARGET_MISSING:" + path)
                for prim in Usd.PrimRange(target):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    sem = Semantics.SemanticsAPI.Apply(prim, "BBoardVisualTarget")
                    sem.CreateSemanticTypeAttr().Set("class")
                    sem.CreateSemanticDataAttr().Set("bboard_visual_target")
                    self.mesh_paths.append(str(prim.GetPath()))
        if not self.mesh_paths:
            raise RuntimeError("BBOARD_TARGET_HAS_NO_MESH")
        product = collector.cameras[0].get_render_product_path()
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.seg = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast",
                                                       init_params={"colorize": False})
        self.rgb.attach([product]); self.seg.attach([product])
        self.frame_index = 0
        self.last_rgb = None
        import json
        (self.evidence/"target_binding.json").write_text(json.dumps({
            "camera": str(product), "mesh_paths": self.mesh_paths,
            "model_resolution": [224,224], "asset_files_modified": False,
            "mask_source": "instance_segmentation_fast"}, indent=2), encoding="utf-8")

    def read(self):
        # 调用者刚完成 policy_observation 的 render；两个 annotator 属于同一渲染产品。
        rgb = self.rgb.get_data()
        segment = self.seg.get_data()
        if isinstance(rgb, dict):
            rgb = rgb.get("data")
        rgb = np.asarray(rgb)
        data = np.asarray(segment.get("data")) if isinstance(segment, dict) else np.asarray(None)
        if rgb.ndim != 3 or data.ndim != 2 or rgb.shape[:2] != data.shape:
            raise RuntimeError("BBOARD_ANNOTATOR_NOT_READY_OR_RGB_MASK_MISMATCH")
        labels = segment.get("info", {}).get("idToLabels")
        if not isinstance(labels, dict):
            raise RuntimeError("BBOARD_SEGMENTATION_LABEL_MAP_MISSING")
        ids = []
        for key, path in labels.items():
            if isinstance(path, str) and any(path == p or path.startswith(p + "/") for p in self.mesh_paths):
                ids.append(int(key))
        mask = np.isin(data, ids)
        frame = np.asarray(Image.fromarray(rgb[...,:3].astype(np.uint8)).resize((224,224), Image.Resampling.BILINEAR))
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((224,224), Image.Resampling.NEAREST), dtype=bool)
        self.frame_index += 1
        self.last_rgb = frame.copy()
        return frame, {"bboard_visual_mask": pack_mask(mask),
                       "bboard_frame_index": self.frame_index,
                       "bboard_evidence_dir": str(self.evidence)}
