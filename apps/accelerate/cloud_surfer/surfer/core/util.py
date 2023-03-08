from mnemonic import Mnemonic


class RandomGenerator:
    def __init__(self, separator="-"):
        self.__generator = Mnemonic(language="English")
        self.separator = separator

    def random_mnemonic(self, n_words=3) -> str:
        words = self.__generator.generate(strength=128).split(" ")[:n_words]
        return self.separator.join(words)
