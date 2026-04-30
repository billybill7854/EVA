"""EVATokenizer — Character-level tokenization with source tagging.

Special tokens encode the source of each message (scaffold, human,
ancestor, self) so EVA always knows who is speaking. This is a
Covenant requirement: no layer pretends to be another.
"""

from __future__ import annotations

from typing import Optional

# Special token IDs
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SELF_ID = 4
HUMAN_ID = 5
SCAFFOLD_ID = 6
ANCESTOR_ID = 7

# Source tag mapping
SOURCE_TOKEN_MAP: dict[str, int] = {
    "self": SELF_ID,
    "human": HUMAN_ID,
    "scaffold": SCAFFOLD_ID,
    "ancestor": ANCESTOR_ID,
}


class EVATokenizer:
    """Character-level (byte-level) tokenizer with source tagging.

    Fixed special tokens: <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3,
    <SELF>=4, <HUMAN>=5, <SCAFFOLD>=6, <ANCESTOR>=7.

    Initial vocab: special tokens + all printable ASCII (32-126).
    Vocabulary can grow during development via add_token().
    """

    def __init__(self) -> None:
        # Initialize token-to-id and id-to-token mappings
        self._token_to_id: dict[str, int] = {
            "<PAD>": PAD_ID,
            "<UNK>": UNK_ID,
            "<BOS>": BOS_ID,
            "<EOS>": EOS_ID,
            "<SELF>": SELF_ID,
            "<HUMAN>": HUMAN_ID,
            "<SCAFFOLD>": SCAFFOLD_ID,
            "<ANCESTOR>": ANCESTOR_ID,
        }
        self._id_to_token: dict[int, str] = {
            v: k for k, v in self._token_to_id.items()
        }

        # Add printable ASCII characters (32-126)
        next_id = len(self._token_to_id)
        for code in range(32, 127):
            char = chr(code)
            if char not in self._token_to_id:
                self._token_to_id[char] = next_id
                self._id_to_token[next_id] = char
                next_id += 1

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self._token_to_id)

    def encode(
        self, text: str, source: Optional[str] = None
    ) -> list[int]:
        """Tokenize text into token IDs.

        Args:
            text: Input text string.
            source: Optional source tag ("self", "human", "scaffold",
                    "ancestor"). If given, prepend appropriate source token.

        Returns:
            List of token IDs.
        """
        ids: list[int] = [BOS_ID]

        # Prepend source token if specified
        if source is not None:
            source_lower = source.lower()
            if source_lower in SOURCE_TOKEN_MAP:
                ids.append(SOURCE_TOKEN_MAP[source_lower])

        # Tokenize character by character
        for char in text:
            if char in self._token_to_id:
                ids.append(self._token_to_id[char])
            else:
                ids.append(UNK_ID)

        ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text string (special tokens are excluded).
        """
        special_ids = {PAD_ID, UNK_ID, BOS_ID, EOS_ID, SELF_ID,
                       HUMAN_ID, SCAFFOLD_ID, ANCESTOR_ID}
        chars: list[str] = []
        for token_id in ids:
            if token_id in special_ids:
                continue
            token = self._id_to_token.get(token_id)
            if token is not None:
                chars.append(token)
        return "".join(chars)

    def add_token(self, token_str: str) -> int:
        """Add a new token to the vocabulary.

        For vocabulary growth during development.

        Args:
            token_str: The token string to add.

        Returns:
            The ID assigned to the new token.
        """
        if token_str in self._token_to_id:
            return self._token_to_id[token_str]

        new_id = len(self._token_to_id)
        self._token_to_id[token_str] = new_id
        self._id_to_token[new_id] = token_str
        return new_id

    def get_source_tag(self, ids: list[int]) -> Optional[str]:
        """Extract the source tag from a token ID sequence.

        Args:
            ids: Token ID sequence.

        Returns:
            Source name string or None if no source token found.
        """
        id_to_source = {v: k for k, v in SOURCE_TOKEN_MAP.items()}
        for token_id in ids:
            if token_id in id_to_source:
                return id_to_source[token_id]
        return None
