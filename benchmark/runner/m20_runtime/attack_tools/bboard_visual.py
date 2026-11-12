"""B 榜局部视觉攻击：掩码来自资产网格，只修改模型前视图像。"""
import base64
import json
from pathlib import Path
import numpy as np

def decode_mask(value, shape):
    if not isinstance(value, dict) or tuple(value.get("shape", ())) != tuple(shape):
        raise ValueError("BBOARD_MASK_MISSING_OR_WRONG_SHAPE")
    bits = np.frombuffer(base64.b64decode(value["bits"], validate=True), dtype=np.uint8)
    count = int(np.prod(shape))
    if len(bits) != (count + 7) // 8:
        raise ValueError("BBOARD_MASK_LENGTH")
    return np.unpackbits(bits, count=count).reshape(shape).astype(bool)

def pack_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    return {"shape": list(mask.shape),
            "bits": base64.b64encode(np.packbits(mask).tobytes()).decode("ascii")}

def recolor(image, mask, mode):
    """输出与输入形状一致；掩码外逐像素不变。灰化在线性 RGB 中保持亮度。"""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("BBOARD_EXPECTS_UINT8_RGB")
    selected = np.asarray(mask, dtype=bool).copy()
    if selected.shape != image.shape[:2]:
        raise ValueError("BBOARD_MASK_SHAPE")
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if mode == "yellow_to_gray":
        selected &= (r > b + 12) & (g > b + 12) & (r >= .85*g) & (r < 2.3*g + 1)
    elif mode == "brown_to_gray":
        selected &= (r > g + 3) & (g > b + 3)
    elif mode == "black_preserve_boundary":
        # 保留目标轮廓的一像素边界，避免把黑色区域扩散到背景。
        padded = np.pad(selected, 1, constant_values=False)
        selected &= padded[:-2,1:-1] & padded[2:,1:-1] & padded[1:-1,:-2] & padded[1:-1,2:]
    elif mode != "surface_to_gray":
        raise ValueError("UNKNOWN_BBOARD_VISUAL_MODE:" + mode)
    out = image.copy()
    if mode == "black_preserve_boundary":
        out[selected] = 0
    elif mode == "surface_to_gray":
        # 黑色本身无彩度，单纯去饱和不会变亮；本地实现灰色值公开设为 128。
        out[selected] = 128
    else:
        srgb = rgb / 255.
        linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
        y = linear @ np.array([.2126, .7152, .0722], np.float32)
        gray = np.where(y <= .0031308, y * 12.92, 1.055 * y ** (1/2.4) - .055)
        gray = np.rint(np.clip(gray, 0, 1) * 255).astype(np.uint8)
        out[selected] = np.repeat(gray[..., None], 3, axis=2)[selected]
    return out, selected

def picture_bboard_target(picture, parameters, context):
    runtime = context.get("runtime", {})
    image = np.asarray(picture)
    mask = decode_mask(runtime.get("bboard_visual_mask"), image.shape[:2])
    out, selected = recolor(image, mask, parameters["mode"])
    folder = runtime.get("bboard_evidence_dir")
    if folder:
        from PIL import Image
        root = Path(folder); root.mkdir(parents=True, exist_ok=True)
        serial = int(runtime["bboard_frame_index"])
        prefix = f"{serial:06d}"
        for suffix, pixels in [("clean", image), ("attacked", out),
                               ("mask", selected.astype(np.uint8)*255)]:
            Image.fromarray(pixels).save(root / (prefix + "_" + suffix + ".png"))
        changed = np.any(image != out, axis=-1)
        record = {"frame": serial, "time_s": runtime.get("time_s"),
                  "mode": parameters["mode"], "target_pixels": int(mask.sum()),
                  "selected_pixels": int(selected.sum()), "changed_pixels": int(changed.sum()),
                  "outside_mask_changes": int((changed & ~mask).sum()),
                  "status": "APPLIED" if changed.any() else "TARGET_NOT_VISIBLE_OR_NO_MATCHING_PIXELS"}
        with (root/"visual_audit.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out
