"""Meta client: cache, budget, retries. No network."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error

from coopx.events import validate
from coopx.meta.client import (BudgetExceeded, CacheMiss, MetaAPIError, MetaClient,
                              model_call_event)


class FakePost:
    def __init__(self, sequence, clock=None):
        self.sequence = list(sequence)
        self.requests = []
        self.clock = clock

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        item = self.sequence.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _ok(text="hello", images=0, inp=10, out=5):
    return 200, {
        "text": text,
        "usage": {
            "input_tokens": inp, "output_tokens": out, "images_processed": images,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }


class MetaClientTests(unittest.TestCase):
    def _client(self, tmp, post, **kwargs):
        env = {"MODEL_API_KEY": "secret-test-key"}
        env.update(kwargs.pop("environ", {}))
        return MetaClient(
            cache_dir=Path(tmp) / "cache", cost_log=Path(tmp) / "cost.jsonl",
            post=post, environ=env, clock=kwargs.pop("clock", None),
            **kwargs)

    def test_cache_hit_replays_latency_and_hides_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            times = iter([0.0, 1.5, 10.0, 10.0])
            post = FakePost([_ok("cached-text")])
            client = self._client(tmp, post, clock=lambda: next(times))
            first = client.respond(
                model="muse-spark-1.3", instructions="x", input="hi",
                query_time_us=1_000, stage="plan")
            self.assertFalse(first.cache_hit)
            self.assertAlmostEqual(first.latency_ms, 1500.0, places=3)
            self.assertEqual(first.availability_time_us, 1_000 + 1_500_000)
            second = client.respond(
                model="muse-spark-1.3", instructions="x", input="hi",
                query_time_us=9_000, stage="plan")
            self.assertTrue(second.cache_hit)
            self.assertEqual(second.latency_ms, first.latency_ms)
            self.assertEqual(second.availability_time_us, 9_000 + 1_500_000)
            self.assertEqual(second.text, "cached-text")
            self.assertEqual(len(post.requests), 1)
            cache_files = list((Path(tmp) / "cache").glob("*.json"))
            blob = Path(tmp).joinpath("cost.jsonl").read_text(encoding="utf-8") + "".join(
                p.read_text(encoding="utf-8") for p in cache_files)
            self.assertNotIn("secret-test-key", blob)
            event = model_call_event(second, "reasoner")
            validate(event)
            self.assertTrue(event["detail"]["cache_hit"])
            self.assertEqual(event["t_replay_us"], event["detail"]["availability_time_us"])

    def test_offline_miss_and_budget_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp, FakePost([]), offline=True)
            with self.assertRaises(CacheMiss):
                client.respond(model="sam-3.1", instructions="s", input="p",
                               query_time_us=0, stage="segment")
            spender = FakePost([_ok("x", images=1)])
            rich = self._client(tmp, spender, budget_usd=0.0)
            with self.assertRaises(BudgetExceeded):
                rich.respond(model="sam-3.1", instructions="s",
                             input=[{"role": "user", "content": [{"type": "input_text", "text": "p"}]}],
                             query_time_us=0, stage="segment")

    def test_5xx_retries_4xx_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            err500 = urllib.error.HTTPError(
                "https://api.meta.ai/v1/responses", 500, "boom", hdrs=None,
                fp=io.BytesIO(b"{}"))
            post = FakePost([err500, _ok("recovered")])
            client = self._client(tmp, post)
            result = client.respond(model="muse-spark-1.3", instructions="x",
                                    input="hi", query_time_us=0, stage="plan")
            self.assertEqual(result.text, "recovered")
            self.assertEqual(len(post.requests), 2)
            err400 = urllib.error.HTTPError(
                "https://api.meta.ai/v1/responses", 400, "nope", hdrs=None,
                fp=io.BytesIO(b"{}"))
            once = FakePost([err400, _ok("should-not-run")])
            bad = self._client(tmp, once)
            with self.assertRaises(MetaAPIError) as ctx:
                bad.respond(model="muse-spark-1.3", instructions="other",
                            input="different", query_time_us=0, stage="plan")
            self.assertEqual(ctx.exception.status, 400)
            self.assertEqual(len(once.requests), 1)
            self.assertNotIn("secret-test-key", str(ctx.exception))

    def test_rejects_contributor_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp, FakePost([]))
            with self.assertRaises(ValueError):
                client.respond(model="muse-spark-1.3-contributor", instructions="x",
                               input="hi", query_time_us=0, stage="plan")


if __name__ == "__main__":
    unittest.main()
