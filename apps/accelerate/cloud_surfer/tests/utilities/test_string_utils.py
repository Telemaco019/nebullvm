import unittest

from surfer.utilities.string_utils import RandomGenerator


class TestRandomGenerator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = RandomGenerator()

    def test_random_mnemonic__generated_words_all_different(self):
        generated = []
        for _ in range(5):
            generated.append(self.generator.random_mnemonic())
        self.assertEqual(len(generated), len(set(generated)))

    def test_random_mnemonic__n_words(self):
        n_words = 5
        mnemonic = self.generator.random_mnemonic(n_words=n_words)
        self.assertEqual(
            n_words, len(mnemonic.split(self.generator.separator))
        )
