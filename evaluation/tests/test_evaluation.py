import base64
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import evaluate_offline_metrics as offline
from evaluation import evaluate_with_vlm_judge as judge
from evaluation.metrics.base_metric import AverageMeter


class EvaluationTests(unittest.TestCase):
    def test_judge_serializes_configured_endpoint_and_png(self):
        requests = []

        def post(url, **kwargs):
            requests.append((url, kwargs))
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "{\"overall_score\": 0.75}"}}]},
            )

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.png"
            image.write_bytes(b"image fixture")
            with patch.dict(sys.modules, {"requests": SimpleNamespace(post=post)}):
                client = judge.VLMJudgeClient(api_key="test-key", base_url="https://example.invalid/v1/")
                output = client.multimodal_conversation(
                    [{"data_type": "text", "data_value": "Score this image"},
                     {"data_type": "image", "data_value": str(image)}],
                    "test-judge", {"max_new_tokens": 2048},
                )

        self.assertEqual(output, ['{"overall_score": 0.75}'])
        self.assertEqual(len(requests), 1)
        url, options = requests[0]
        self.assertEqual(url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(options["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(options["json"]["model"], "test-judge")
        self.assertEqual(options["json"]["max_completion_tokens"], 2048)
        image_url = options["json"]["messages"][0]["content"][1]["image_url"]["url"]
        prefix, content = image_url.split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(base64.b64decode(content), b"image fixture")

    def test_judge_uses_environment_and_allows_explicit_overrides(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key", "OPENAI_BASE_URL": "https://env.invalid/v1"}), \
             patch.dict(sys.modules, {"requests": SimpleNamespace()}):
            client = judge.VLMJudgeClient()
            self.assertEqual(client.url, "https://env.invalid/v1/chat/completions")
            self.assertEqual(client.headers["Authorization"], "Bearer env-key")
            client = judge.VLMJudgeClient(api_key="arg-key", base_url="https://arg.invalid/v1")
            self.assertEqual(client.url, "https://arg.invalid/v1/chat/completions")
            self.assertEqual(client.headers["Authorization"], "Bearer arg-key")

    def test_missing_judge_configuration_fails_before_loading_transport(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                judge.VLMJudgeClient()
            with self.assertRaisesRegex(ValueError, "OPENAI_BASE_URL"):
                judge.VLMJudgeClient(api_key="test-key")

    def test_judge_parses_score_response_and_preserves_clamping(self):
        parsed = judge.parse_score_json('```json\n{"reason": "Clear shapes", "overall_score": 0.75}\n```')
        self.assertEqual(parsed, {"reason": "Clear shapes", "overall_score": 0.75})
        self.assertEqual(judge.extract_overall_score(parsed), 0.75)
        self.assertEqual(judge.extract_overall_score({"overall_score": 2}), 1)
        self.assertEqual(judge.extract_overall_score({"overall_score": -1}), 0)
        self.assertIsNone(judge.parse_score_json("no score"))
        self.assertIsNone(judge.extract_overall_score({}))

    def test_loaders_preserve_successful_inference_sample_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.png"
            image.touch()
            sample = {"id": "sample-1", "type": "icon", "prompt": "A blue star", "png_path": str(image)}
            results = Path(directory) / "results.json"
            results.write_text(json.dumps({"samples": [
                dict(sample, status="success"),
                dict(sample, status="failed"),
                dict(sample, status="skipped_existing"),
                dict(sample, status="success", png_path=str(Path(directory) / "missing.png")),
            ]}))
            expected = {
                "icon": {"paths": [str(image)], "prompts": ["A blue star"], "ids": ["sample-1"]},
                "illustration": {"paths": [], "prompts": [], "ids": []},
            }
            self.assertEqual(offline.load_results(results), expected)
            self.assertEqual(judge.load_results(results), expected)

    def test_metric_average_weights_batch_sizes_and_resets(self):
        meter = AverageMeter()
        meter.update(0.5, 2)
        meter.update(1.0, 1)
        self.assertAlmostEqual(meter.avg, 2 / 3)
        self.assertEqual(meter.count, 3)
        meter.reset()
        self.assertEqual(meter.avg, 0)
        self.assertEqual(meter.count, 0)


if __name__ == "__main__":
    unittest.main()
