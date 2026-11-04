"""S4b wiring that does not call the network."""
import unittest

from coopx.pipeline import range_contradicts_callback


class RangeCallbackTests(unittest.TestCase):
    def test_fail_closed(self):
        callback = range_contradicts_callback(None, None, {"us": 0})
        self.assertFalse(callback("a", "b"))
        self.assertFalse(callback(1, "b"))
        broken = range_contradicts_callback(object(), object(), {"us": 0})
        self.assertFalse(broken("aa", "bb"))


if __name__ == "__main__":
    unittest.main()
