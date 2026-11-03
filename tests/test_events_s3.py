"""Contract 5 event shapes (query, answer, verdict, model_call)."""
import copy
import unittest

from coopx import events

REF = "a" * 64


def ev(type_, **kw):
    base = {"schema": events.SCHEMA, "t_replay_us": 1_000, "type": type_, "source_id": "reasoner",
            "dest_id": None, "message_id": "q1", "message_type": None, "evidence_ref": None,
            "bytes": None, "check_codes": [], "state": None, "detail": {}}
    base.update(kw)
    return base


GOOD = {
    "query": ev("query", dest_id="rsu-2", message_type="query", detail={
        "tier": "T3", "region": {"center": [1.0, 2.0], "radius_m": 5.0}, "time_us": 900,
        "concept": "person", "reason": "covers the region"}),
    "answer": ev("answer", source_id="rsu-2", dest_id="reasoner", message_type="answer",
                 state="detected", detail={"observed_time_us": 900, "coverage_ref": None,
                                           "evidence_refs": [REF]}),
    "verdict": ev("verdict", message_id="v1", message_type="verdict", state="corroborated",
                  detail={"subject": "region-1", "evidence_refs": [REF], "reason": "x",
                          "model_call_ids": ["c1"]}),
    "model_call": ev("model_call", message_id="c1", t_replay_us=2_900, detail={
        "product": "muse-spark-1.3", "stage": "plan", "cache_hit": False, "latency_ms": 1900.0,
        "input_tokens": 10, "output_tokens": 5, "images": 0, "cost_usd": 0.0001,
        "query_time_us": 1_000, "availability_time_us": 2_900, "precomputed": False}),
}


class S3EventTests(unittest.TestCase):
    def test_good_events_validate(self):
        for event in GOOD.values():
            events.validate(event)

    def test_bad_events_rejected(self):
        mutations = [
            ("query", lambda e: e["detail"].__setitem__("tier", "T9")),
            ("query", lambda e: e["detail"]["region"].__setitem__("radius_m", 0)),
            ("answer", lambda e: e.__setitem__("state", "free")),
            ("answer", lambda e: e["detail"].__setitem__("evidence_refs", [])),
            ("answer", lambda e: (e.__setitem__("state", "observed_free"))),
            ("verdict", lambda e: e["detail"].__setitem__("evidence_refs", [])),
            ("verdict", lambda e: e.__setitem__("check_codes", ["OK"])),
            ("model_call", lambda e: e.__setitem__("t_replay_us", 1_000)),
            ("model_call", lambda e: e["detail"].__setitem__("product", "muse-spark-1.3-contributor")),
        ]
        for kind, mutate in mutations:
            event = copy.deepcopy(GOOD[kind])
            mutate(event)
            with self.assertRaises(ValueError, msg=kind):
                events.validate(event)


if __name__ == "__main__":
    unittest.main()


class MapObserverPoseTests(unittest.TestCase):
    def snap(self, observers):
        return {"schema": events.SCHEMA, "t_replay_us": 0, "type": "map_snapshot",
                "source_id": "ego", "dest_id": None, "message_id": None, "message_type": None,
                "evidence_ref": None, "bytes": None, "check_codes": [], "state": None,
                "detail": {"occupied": [], "free": [], "contested": [], "unknown_count": 3600,
                           "observers": observers}}

    def test_observer_poses(self):
        events.validate(self.snap([{"id": "rsu-2", "xy": [21.5, -180.0]}]))
        for bad in ([{"id": "a", "xy": [1, float("nan")]}], [{"id": "a", "xy": [1]}],
                    [{"id": "a", "xy": [1, 2]}, {"id": "a", "xy": [3, 4]}], [{"id": "", "xy": [1, 2]}]):
            with self.assertRaises(ValueError):
                events.validate(self.snap(bad))
