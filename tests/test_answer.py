import unittest

from gatedtrain_pack_vlm_clevr import answer


class AnswerTests(unittest.TestCase):
    def test_canonical(self):
        table = [
            ("<answer>3</answer>", "3"),
            ("so it is 4 <answer> three </answer>", "3"),
            ("<answer>1</answer> then <answer>07</answer>", "7"),
            ("<think>9</think>yes", "yes"),
            ("answer: Yes.", "yes"),
            ("there are twelve", "12"),
            ("nothing to see here", None),
            ("", None),
            ("2 then 5 then no", "no"),
            ("<think>a</think>b</think>8", "8"),
        ]
        for text, want in table:
            with self.subTest(text=text):
                self.assertEqual(answer.canonical(text), want)

    def test_non_ascii_digits_are_not_numbers(self):
        self.assertIsNone(answer.canonical("\u0663"))
        self.assertFalse(answer.label_ok("\u0663"))
        self.assertEqual(answer.canonical("twenty ten 10"), "10")
        self.assertEqual(answer.canonical("<answer>99999999999999999999</answer>"), "99999999999999999999")

    def test_untagged_vs_tagged(self):
        self.assertEqual(answer.canonical("<answer>1</answer> and 2"), "1")

    def test_score(self):
        self.assertEqual(answer.score("<answer>3</answer>", "3"), 1.0)
        self.assertEqual(answer.score("<answer>3</answer>", "03"), 1.0)
        self.assertEqual(answer.score("yes.", "Yes"), 1.0)
        self.assertEqual(answer.score("<answer>4</answer>", "3"), 0.0)
        self.assertEqual(answer.score("nothing", "3"), 0.0)
        self.assertEqual(answer.score(None, "3"), 0.0)
        self.assertEqual(answer.score(5, "5"), 0.0)

    def test_score_bad_label_never_raises(self):
        self.assertEqual(answer.score("3", ""), 0.0)
        self.assertEqual(answer.score("3", "3 4"), 0.0)

    def test_label_ok(self):
        for good in ("3", "03", "Yes", "no", "twelve", " 7 "):
            with self.subTest(good=good):
                self.assertTrue(answer.label_ok(good))
        for bad in ("", "maybe", "3 4", "-1", 5, None, "3.5"):
            with self.subTest(bad=bad):
                self.assertFalse(answer.label_ok(bad))

    def test_canonical_label(self):
        self.assertEqual(answer.canonical_label("Twelve"), "12")
        self.assertIsNone(answer.canonical_label("x"))


if __name__ == "__main__":
    unittest.main()
