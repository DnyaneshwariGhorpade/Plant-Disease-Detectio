"""Diagnosis knowledge base lookup.

The knowledge base is not duplicated here: it is read from the sibling CNN
project (``data/diagnosis.json``) unless a local copy exists, so both projects
describe the 15 classes with exactly the same text.
"""

import json
import os

from preprocessing import resolve_diagnosis_path

DIAGNOSIS_PATH = resolve_diagnosis_path()


def load_diagnosis(path=None):
    path = path or DIAGNOSIS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_diagnosis(class_name, confidence=None, path=None):
    data = load_diagnosis(path)
    entry = data.get(class_name)
    if entry is None:
        return None
    result = dict(entry)
    result["class_name"] = class_name
    if confidence is not None:
        result["confidence"] = float(confidence)
    return result
