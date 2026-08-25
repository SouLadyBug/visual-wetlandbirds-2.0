from __future__ import annotations

import ast
import re
from typing import Any


def parse_bbox_cell(value: Any) -> list[dict[str, Any]]:
    """Parse the annotation formats encountered in the supplied CSV files."""
    if value is None:
        return []
    if isinstance(value, float) and value != value:
        return []
    if isinstance(value, (list, tuple, dict)):
        obj = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            obj = text

    if isinstance(obj, dict):
        obj = [obj]
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            if isinstance(item, dict):
                coords = None
                for key in ("box", "bbox", "xyxy"):
                    if key in item:
                        coords = item[key]
                        break
                if coords is None:
                    keys = ("x1", "y1", "x2", "y2")
                    if all(k in item for k in keys):
                        coords = [item[k] for k in keys]
                if coords is not None and len(coords) == 4:
                    result.append({"bird_id": item.get("bird_id", item.get("id")), "box": tuple(map(float, coords))})
            else:
                nums = re.findall(r"-?\d+(?:\.\d+)?", str(item))
                if len(nums) >= 4:
                    result.append({"bird_id": None, "box": tuple(map(float, nums[:4]))})
        return result

    nums = re.findall(r"-?\d+(?:\.\d+)?", str(obj))
    return [
        {"bird_id": None, "box": tuple(map(float, nums[i:i + 4]))}
        for i in range(0, len(nums) - 3, 4)
    ]
