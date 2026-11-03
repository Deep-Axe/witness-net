"""SAM parser on the recorded probe token and synthetic strings. No network."""
import unittest

from coopx.meta.sam import parse_sam_text

PROBE = "<0f>0<|box;x1=32;y1=32;x2=95;y2=95;w=128;h=128|>"


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


if __name__ == "__main__":
    unittest.main()
