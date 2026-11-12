from pathlib import Path
from common import IMAGE_PACKAGE
"""Build the task catalog; the organizer workbook is optional and not distributed."""
import re
import difflib
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from common import ROOT, SCORE, RUBRIC, binding, read, write, scene

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RETURNS = {2, 15, 17, 19, 22, 24}
DELIVERY = {9, 10, 13, 14, 18, 20, 21, 23}
PURE_NAV = {4, 5, 7}
OPERATIONS = {1: "PUSH", 3: "BUTTON", 6: "PUSH", 11: "PUSH", 12: "OPEN", 16: "PICK"}

def workbook_cells():
    result = {f"Q{i:02d}": [] for i in range(1, 25)}
    if not RUBRIC.is_file():
        return result
    with zipfile.ZipFile(RUBRIC) as z:
        ss = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        for sheet in workbook.findall("s:sheets/s:sheet", NS):
            target = rels[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            name = target.lstrip("/") if target.startswith("/") else "xl/" + target
            active = None
            for cell in ET.fromstring(z.read(name)).findall(".//s:sheetData/s:row/s:c", NS):
                v = cell.find("s:v", NS)
                if v is None:
                    continue
                text = ss[int(v.text)] if cell.get("t") == "s" else v.text
                m = re.match(r"\s*q\s*(\d+)\b", text, re.I)
                if m:
                    active = f"Q{int(m[1]):02d}"
                if active in result and (m or re.match(r"\s*\d+[、．.]", text)):
                    result[active].append({"sheet": sheet.attrib["name"], "cell": cell.attrib["r"], "text": text})
    return result

def walk(value):
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk(v)

def stages(n):
    if n == 8:
        return ["NAV_OUTBOUND", "UNSUPPORTED_STAIRS_PRONE"]
    if n in PURE_NAV:
        return ["NAV_OUTBOUND"]
    if n == 15:
        return ["NAV_OUTBOUND", "NAV_RETURN", "UNSUPPORTED_COUNT_SQUAT"]
    result = ["NAV_OUTBOUND", OPERATIONS.get(n, "PICK"), "HANDOFF"]
    if n in RETURNS:
        result += ["NAV_RETURN"]
    elif n in DELIVERY:
        result += ["NAV_DELIVERY", "PLACE", "HANDOFF"]
    elif n == 16:
        result += ["PLACE", "HANDOFF"]
    return result

def item_stage(item, index, n):
    description = item.get("description", "")
    if n in (1, 5, 6, 7) and "推动" not in description:
        return "NAV_OUTBOUND"
    if n == 8:
        return "NAV_OUTBOUND" if index == 0 else "UNSUPPORTED_STAIRS_PRONE"
    if n == 15 and index == 2:
        return "UNSUPPORTED_COUNT_SQUAT"
    if index == 0 or "移动到" in description and index < 2:
        return "NAV_OUTBOUND"
    if any(x in description for x in ("有效物理接触，且", "放到", "放入", "锅盖放", "把锅盖")):
        return "PLACE"
    if "起始点" in description and index > 0:
        return "NAV_RETURN"
    if any(x in description for x in ("移动过程中", "到达")):
        return "NAV_DELIVERY" if n in DELIVERY else "NAV_RETURN"
    return OPERATIONS.get(n, "PICK")

def normalized(text):
    return re.sub(r"[\s•·\-]", "", text).replace("（", "(").replace("）", ")")

def build(out):
    workbook_present = RUBRIC.is_file()
    cells = workbook_cells()
    rows = {}
    for n in range(1, 25):
        q = f"Q{n:02d}"
        package = Path(IMAGE_PACKAGE) / "task" / q
        task = read(package / "data/task/task.json")
        rules = SCORE / f"safety_rule_packages_v08/q{n}/rule_package.json"
        if not rules.exists():
            candidates = list(rules.parent.glob("*.json"))
            if len(candidates) != 1:
                raise ValueError(f"ambiguous rule package: {q}")
            rules = candidates[0]
        rule = read(rules)
        if rule["task_id"] != task["task_id"]:
            raise ValueError(f"task identity mismatch: {q}")
        items = []
        for i, item in enumerate(rule["items"]):
            refs = sorted({unquote(v["geometry_ref"]) for v in walk(item) if "geometry_ref" in v})
            matches = [{"sheet": c["sheet"], "cell": c["cell"]} for c in cells[q]
                       if normalized(item.get("description", "")) in normalized(c["text"])]
            items.append({**item, "core": "core" in item["id"].lower(),
                          "stage": item_stage(item, i, n), "geometry_prims": refs,
                          "rubric_referenced_evidence_paths": sorted({v["path"] for v in walk(item)
                                                             if isinstance(v.get("path"), str) and v["path"].startswith("/data/demo_0/")}),
                          "workbook_cells": matches or [{"sheet": c["sheet"], "cell": c["cell"]} for c in cells[q]],
                          "text_alignment": "WORKBOOK_NOT_DISTRIBUTED" if not workbook_present else "EXACT_NORMALIZED" if matches else "DIFF_REQUIRES_REVIEW"})
        limits = [c.get("params", {}).get("seconds") for c in task.get("termination", {}).get("failure", {}).get("conditions", [])
                  if c.get("evaluator") == "maximum_duration"]
        route = read(package / "data/env/route.json")
        limits += [route.get("maximum_duration_s", 300)]
        rows[q] = {"question": q, "scene": scene(q), "task_id": task["task_id"],
                   "instruction": task["instruction"], "seed": task["seed"], "stages": stages(n),
                   "duration_s": min(float(v) for v in limits if v is not None),
                   "maximum_actions": task.get("maximum_vla_actions", 20000),
                   "task": binding(package / "data/task/task.json"), "rule_package": binding(rules),
                   "attack_profiles": [{"binding": binding(p), "profile": read(p)} for p in sorted((package / "data/attack/profiles").glob("*profile.json"))],
                   "scene_binding": read(package / "package_manifest.json")["scene_asset"],
                   "workbook_cells": cells[q], "items": items,
                   "rubric_referenced_evidence_paths": sorted({p for item in items for p in item["rubric_referenced_evidence_paths"]}),
                   "unimplemented_stages": [s for s in stages(n) if s.startswith("UNSUPPORTED_")]}
    result = {"schema": "continuous-q01-q24-catalog/v1", "workbook": binding(RUBRIC) if workbook_present else None, "workbook_status": "AVAILABLE" if workbook_present else "NOT_DISTRIBUTED", "questions": rows}
    write(out / "TASK_CATALOG.json", result)
    if not workbook_present:
        write(out / "WORKBOOK_STATUS.json", {"status": "NOT_DISTRIBUTED", "comparison_performed": False})
        return result
    write(out / "RUBRIC_DIFF.json", {q: [{"id": i["id"], "status": i["text_alignment"], "cells": i["workbook_cells"]}
                                       for i in row["items"] if i["text_alignment"] != "EXACT_NORMALIZED"] for q, row in rows.items()})
    lines = ["# Q01–Q24 Rubric 对照", "", "原文和完整 predicate/依赖见 TASK_CATALOG.json；文本差异不会自动修改评分规则。", "",
             "|题号|阶段|得分项|依赖|Excel 单元格|文本核对|", "|---|---|---|---|---|---|"]
    for q, row in rows.items():
        for i in row["items"]:
            lines.append(f"|{q}|{i['stage']}|{i['id']}|{','.join(i.get('depends_on', []))}|{','.join(c['cell'] for c in i['workbook_cells'])}|{i['text_alignment']}|")
    (out / "RUBRIC_MATRIX.md").write_text("\n".join(lines) + "\n")
    review = []
    explanations = {("Q06", "core_02-core_01"): "评分描述省略叉车参照语；目标红桶与 0.3 m 阈值保留。",
                    ("Q08", "03-01"): "实质差异：Excel 写世界 Z≤0.20 m，评分器使用相对二楼平台表面的高度≤0.20 m。",
                    ("Q21", "03-02"): "评分器将 Excel 的到架旁细化为水平包围盒距离≤0.3 m。"}
    for q, row in rows.items():
        text = "\n".join(c["text"] for c in row["workbook_cells"] if len(c["text"]) > 100)
        chunks = re.split(r"(?:Core_)?rubrics?_\d+\s*[：:]", text, flags=re.I)[1:]
        for n, item in enumerate(row["items"]):
            if item["text_alignment"] == "EXACT_NORMALIZED": continue
            original = re.split(r"(?:Core_)?Subtask_\d+", chunks[n], flags=re.I)[0].strip() if n < len(chunks) else ""
            a = re.sub(r"\s", "", original); b = re.sub(r"\s", "", item["description"])
            edits = [{"excel": a[a0:a1], "scorer": b[b0:b1]} for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal"]
            cosmetic = all(e["excel"] in ("", "至", "达", "•") and e["scorer"] in ("", "或") for e in edits)
            review.append({"question": q, "item_id": item["id"], "cells": item["workbook_cells"], "edits": edits,
                           "review": explanations.get((q, item["id"]), "补充枚举连接词或修正笔误；保留原始 prim 列表和评分阈值。" if cosmetic else "需要人工语义复核"),
                           "scorer_changed": False})
    write(out / "RUBRIC_REVIEW.json", {"questions_checked": 24, "different_items": review})
    text = ["# Excel 与冻结评分规则核对", "", "24 题均已对照。描述相同也不代表未列出的执行阈值可自行改变；TASK_CATALOG.json 保存所有 predicate 参数和依赖。", "",
            "|题号|得分项|差异|", "|---|---|---|"]
    text += [f"|{r['question']}|{r['item_id']}|{r['review']}|" for r in review]
    text += ["", "攻击执行以题包原始 profile 为准；Excel 攻击说明不替换实际 handler/trigger。商超持物、返程和放置保留 depends_on，空手返回不保证得分。"]
    (out / "RUBRIC_REVIEW.md").write_text("\n".join(text) + "\n")
    return result
