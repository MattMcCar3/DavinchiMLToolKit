from davinchi.video.timeline.segments import build_timeline
from davinchi.video.timeline.export import export_timeline_json
import json


def test_build_timeline_shape():
    timeline = build_timeline(
        duration_sec=30.0,
        cuts=[10.0, 20.0],
        silence_spans=[(4.0, 8.0)],
        stage_labels=[(2.0, "stage_a")],
    )
    assert timeline["duration_sec"] == 30.0
    assert timeline["cuts"] == [10.0, 20.0]
    assert timeline["silence_spans"] == [{"start": 4.0, "end": 8.0}]
    assert timeline["stage_labels"] == [{"timestamp": 2.0, "label": "stage_a"}]


def test_build_timeline_sorts_cuts():
    timeline = build_timeline(duration_sec=10.0, cuts=[5.0, 1.0, 3.0])
    assert timeline["cuts"] == [1.0, 3.0, 5.0]


def test_export_timeline_json_roundtrip(tmp_path):
    timeline = build_timeline(duration_sec=10.0, cuts=[1.0])
    out_path = tmp_path / "timeline.json"
    export_timeline_json(timeline, out_path)

    loaded = json.loads(out_path.read_text())
    assert loaded == timeline
