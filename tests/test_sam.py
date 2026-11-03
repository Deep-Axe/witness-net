"""SAM parser on the recorded probe token and synthetic strings. No network."""
import unittest

from coopx.meta.mask import encode_one_bit
from coopx.meta.sam import parse_sam_text

PROBE = "<0f>0<|box;x1=32;y1=32;x2=95;y2=95;w=128;h=128|>"
GOLD_ONE_BIT = "!!!!!(QO(0lu8?"  # 5x5, documented @meta-sam/parser example


class SamParserTests(unittest.TestCase):
    def test_probe_box_in_native_frame(self):
        detections = parse_sam_text(PROBE, (128, 128))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].box_xyxy, (32.0, 32.0, 95.0, 95.0))
        self.assertIsNone(detections[0].mask)

    def test_rescale_and_multiple_boxes(self):
        text = ("<|box;x1=32;y1=32;x2=95;y2=95;w=128;h=128|>"
                "<|box;x1=0;y1=0;x2=64;y2=64;w=128;h=128|>")
        detections = parse_sam_text(text, (256, 256))
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].box_xyxy, (64.0, 64.0, 190.0, 190.0))
        self.assertEqual(detections[1].box_xyxy, (0.0, 0.0, 128.0, 128.0))

    def test_garbage_is_empty_not_an_error(self):
        self.assertEqual(parse_sam_text("", (64, 64)), [])
        self.assertEqual(parse_sam_text("no tokens here", (64, 64)), [])
        self.assertEqual(parse_sam_text("<|box;x1=bad;y1=0;x2=1;y2=1;w=8;h=8|>", (8, 8)), [])

    def test_one_bit_roundtrip_and_gold_payload(self):
        raster = bytes((i + j) % 2 for i in range(5) for j in range(5))
        payload = encode_one_bit(raster, 5, 5)
        text = (f"<|box;x1=0;y1=0;x2=4;y2=4;w=5;h=5|>"
                f"<|mask;x=0;y=0;data=5,5,{payload}|>")
        detections = parse_sam_text(text, (5, 5))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].mask.shape, (5, 5))
        self.assertEqual(tuple(detections[0].mask.reshape(-1).astype(int)),
                         tuple(raster))
        gold = parse_sam_text(
            f"<|box;x1=0;y1=0;x2=4;y2=4;w=5;h=5|><|mask;x=0;y=0;data=5,5,{GOLD_ONE_BIT}|>",
            (5, 5))
        self.assertEqual(gold[0].mask.shape, (5, 5))
        self.assertTrue(set(gold[0].mask.reshape(-1).tolist()) <= {False, True})

    def test_mask_sits_inside_its_box_after_resize(self):
        raster = bytes([1] * 16)
        payload = encode_one_bit(raster, 4, 4)
        text = (f"<|box;x1=4;y1=4;x2=7;y2=7;w=16;h=16|>"
                f"<|mask;x=0;y=0;data=4,4,{payload}|>")
        mask = parse_sam_text(text, (16, 16))[0].mask
        self.assertEqual(mask.shape, (16, 16))
        self.assertTrue(mask[4:8, 4:8].all())
        self.assertFalse(mask[:4, :].any() or mask[8:, :].any() or mask[:, :4].any() or mask[:, 8:].any())

    def test_garbage_mask_is_none_not_a_crash(self):
        text = "<|box;x1=0;y1=0;x2=3;y2=3;w=8;h=8|><|mask;x=0;y=0;data=2,2,~not-valid|>"
        detections = parse_sam_text(text, (8, 8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].box_xyxy, (0.0, 0.0, 3.0, 3.0))
        self.assertIsNone(detections[0].mask)


if __name__ == "__main__":
    unittest.main()


class SamDocumentedGrammarTests(unittest.TestCase):
    def test_confidence_and_multi_record_line(self):
        text = ("<0f>0<|box;x1=10;y1=20;x2=30;y2=60;w=100;h=80;c=0.91|><|mask;x=0;y=0;c=0.91;data=2,2,~abc|>,"
                "3<|box;x1=50;y1=5;x2=70;y2=40;w=100;h=80;c=0.42|><|mask;x=0;y=0;c=0.42;data=2,2,~abc|>")
        boxes = parse_sam_text(text, (100, 80))
        self.assertEqual([d.box_xyxy for d in boxes], [(10.0, 20.0, 30.0, 60.0), (50.0, 5.0, 70.0, 40.0)])
        self.assertEqual([d.score for d in boxes], [0.91, 0.42])

    def test_stream_collection(self):
        from coopx.meta.client import _collect_stream
        lines = [b'event: x', b'data: {"type":"response.output_text.delta","delta":"<0f>0<|box;"}',
                 b'data: {"type":"response.output_text.delta","delta":"x1=1;y1=2;x2=3;y2=4;w=9;h=9|>"}',
                 b'data: {"type":"response.completed","response":{"usage":{"images_processed":1}}}']
        out = _collect_stream(lines)
        self.assertEqual(len(parse_sam_text(out["text"], (9, 9))), 1)
        self.assertEqual(out["usage"]["images_processed"], 1)
