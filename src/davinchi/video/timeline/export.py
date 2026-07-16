"""Serialize a timeline dict to JSON, for handoff to the Resolve integration
(or for a human to eyeball before trusting it near real footage)."""
import json


def export_timeline_json(timeline, path):
    with open(path, "w") as f:
        json.dump(timeline, f, indent=2)
    return path
