def get_pair(tokens: list[int]):
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


class Tokenizer:
    def __init__(self, input_text: str):
        self.input_text = input_text
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def _normalizer(self, input_text: str):
        return input_text.lower()

    def _utfencode(self, input_text: str):
        return list(input_text.encode("utf-8"))

    def _bpeprocessing(
        self,
        tokens: list[int],
        vocab_size: int,
        min_frequency: int = 2,
    ):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        tokens = tokens.copy()

        for new_id in range(256, vocab_size):
            counts = get_pair(tokens)

            if not counts:
                break

            best_pair = max(counts, key=counts.get)

            if counts[best_pair] < min_frequency:
                break

            tokens = self._merge_pair(tokens, best_pair, new_id)

            self.merges[best_pair] = new_id
            self.vocab[new_id] = (
                self.vocab[best_pair[0]]
                + self.vocab[best_pair[1]]
            )

        return tokens

    def _merge_pair(
        self,
        tokens: list[int],
        pair: tuple[int, int],
        new_id: int,
    ):
        merged_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and (tokens[i], tokens[i + 1]) == pair
            ):
                merged_tokens.append(new_id)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        return merged_tokens

    def _apply_merges(self, tokens: list[int]):
        tokens = tokens.copy()

        while len(tokens) >= 2:
            available_pairs = [
                pair
                for pair in zip(tokens, tokens[1:])
                if pair in self.merges
            ]

            if not available_pairs:
                break

            best_pair = min(
                available_pairs,
                key=lambda pair: self.merges[pair],
            )

            tokens = self._merge_pair(
                tokens,
                best_pair,
                self.merges[best_pair],
            )

        return tokens
    
    ##PUBLIC METHODS EXPOSED OUTSIDE
    
    def train(
        self,
        text: str,
        vocab_size: int,
        min_frequency: int = 2,
    ):
        normalized_text = self._normalizer(text)
        tokens = self._utfencode(normalized_text)

        self._bpeprocessing(
            tokens,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

        return self
    
    def encode(self, text: str) -> list[int]:
        normalized_text = self._normalizer(text)
        tokens = self._utfencode(normalized_text)
        return self._apply_merges(tokens)

    def tokenize(self, text: str) -> list[bytes]:
        token_ids = self.encode(text)
        return [self.vocab[token_id] for token_id in token_ids]
    
    def decode(self, token_ids: list[int]) -> str:
        # look up self.vocab[token_id] give its bytes
        # join bytes to reconstruct full UTF-* byte sequence
        # convert the combined bytes into a python string
        token_bytes = [self.vocab[token_id] for token_id in token_ids]
        combined_bytes = b"".join(token_bytes)

        return combined_bytes.decode("utf-8")