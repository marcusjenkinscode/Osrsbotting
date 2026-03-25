"""OSRS chat-line parser with deduplication and message buffering.

Supported chat formats
----------------------
* ``[Username] Message``              — public chat
* ``[Clan] [Username] Message``       — clan chat
* ``[Guest Clan] [Username] Message`` — guest clan
* ``From [Username]: Message``        — private message (inbound)
* ``To [Username]: Message``          — private message (outbound)
* ``Trade with [Username]``           — trade window
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

ChatType = str  # "public" | "clan" | "guest_clan" | "private_in" | "private_out" | "trade" | "system"


@dataclass
class ParsedMessage:
    """A single parsed OSRS chat line.

    Attributes:
        chat_type: Category of chat message.
        author: In-game username or None for system messages.
        message: Cleaned message body.
        raw_ocr: Raw OCR string before parsing.
        timestamp: Unix epoch when the message was parsed.
    """

    chat_type: ChatType
    author: Optional[str]
    message: str
    raw_ocr: str
    timestamp: float = field(default_factory=time.time)

    def similarity_key(self) -> str:
        """Return a string used for deduplication comparisons."""
        return f"{self.author or ''}:{self.message}"


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

# [Username] Message  — public chat (brackets contain alphanumeric + spaces)
_RE_PUBLIC = re.compile(r"^\[(?P<author>[A-Za-z0-9 _\-]+)\]\s+(?P<message>.+)$")

# [Clan] [Username] Message
_RE_CLAN = re.compile(
    r"^\[Clan\]\s+\[(?P<author>[A-Za-z0-9 _\-]+)\]\s+(?P<message>.+)$", re.IGNORECASE
)

# [Guest Clan] [Username] Message
_RE_GUEST_CLAN = re.compile(
    r"^\[Guest\s+Clan\]\s+\[(?P<author>[A-Za-z0-9 _\-]+)\]\s+(?P<message>.+)$", re.IGNORECASE
)

# From Username: Message   (private inbound)
_RE_PRIVATE_IN = re.compile(r"^From\s+(?P<author>[A-Za-z0-9 _\-]+):\s+(?P<message>.+)$")

# To Username: Message   (private outbound)
_RE_PRIVATE_OUT = re.compile(r"^To\s+(?P<author>[A-Za-z0-9 _\-]+):\s+(?P<message>.+)$")

# Trade with Username
_RE_TRADE = re.compile(r"^Trade\s+with\s+(?P<author>[A-Za-z0-9 _\-]+)$", re.IGNORECASE)

_PATTERNS: List[Tuple[ChatType, re.Pattern]] = [
    ("clan", _RE_CLAN),
    ("guest_clan", _RE_GUEST_CLAN),
    ("private_in", _RE_PRIVATE_IN),
    ("private_out", _RE_PRIVATE_OUT),
    ("trade", _RE_TRADE),
    ("public", _RE_PUBLIC),  # most generic — must be last
]

# ---------------------------------------------------------------------------
# Deduplication buffer
# ---------------------------------------------------------------------------

# How similar two messages must be (0–1) to be considered duplicates
_SIMILARITY_THRESHOLD = 0.85

# Seconds a message lives in the dedup buffer before expiring
_MESSAGE_TTL_S = 30.0

# Maximum entries in the buffer
_BUFFER_MAX = 10


class _MessageBuffer:
    """Circular buffer storing the last N messages for deduplication.

    Args:
        max_size: Maximum number of messages to remember.
        ttl: Time-to-live for each entry in seconds.
    """

    def __init__(self, max_size: int = _BUFFER_MAX, ttl: float = _MESSAGE_TTL_S) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._entries: List[Tuple[float, str]] = []  # (timestamp, similarity_key)

    def is_duplicate(self, msg: ParsedMessage) -> bool:
        """Return True if *msg* is a near-duplicate of a recent message.

        Also evicts expired entries as a side-effect.

        Args:
            msg: Candidate message to check.

        Returns:
            True if a sufficiently similar recent message exists.
        """
        now = time.time()
        self._evict_expired(now)

        key = msg.similarity_key()
        for _ts, stored_key in self._entries:
            ratio = SequenceMatcher(None, key, stored_key).ratio()
            if ratio >= _SIMILARITY_THRESHOLD:
                logger.debug("Duplicate detected (ratio={:.2f}): {!r}", ratio, key)
                return True
        return False

    def add(self, msg: ParsedMessage) -> None:
        """Store *msg* in the buffer, evicting the oldest if at capacity.

        Args:
            msg: Message to remember.
        """
        if len(self._entries) >= self._max_size:
            self._entries.pop(0)
        self._entries.append((msg.timestamp, msg.similarity_key()))

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._ttl
        self._entries = [(ts, k) for ts, k in self._entries if ts > cutoff]


# ---------------------------------------------------------------------------
# Public parser
# ---------------------------------------------------------------------------


class ChatParser:
    """Parse raw OCR text into structured OSRS chat messages.

    Args:
        keyword_filters: If non-empty, only messages containing at least one
            keyword (case-insensitive) are returned.
        ignore_players: Usernames whose messages are always discarded.
        min_confidence: Minimum OCR confidence; messages below this are dropped.

    Example:
        >>> parser = ChatParser()
        >>> msg = parser.parse("[Zezima] Nice bank", raw_ocr="[Zezima] Nice bank", confidence=0.9)
        >>> msg.chat_type
        'public'
    """

    def __init__(
        self,
        keyword_filters: Optional[List[str]] = None,
        ignore_players: Optional[List[str]] = None,
        min_confidence: float = 0.75,
    ) -> None:
        self._keyword_filters = [kw.lower() for kw in (keyword_filters or [])]
        self._ignore_players = {p.lower() for p in (ignore_players or [])}
        self._min_confidence = min_confidence
        self._buffer = _MessageBuffer()

    def parse(
        self, text: str, raw_ocr: str = "", confidence: float = 1.0
    ) -> Optional[ParsedMessage]:
        """Parse a single line of OCR text.

        Args:
            text: Cleaned OCR output line.
            raw_ocr: Raw OCR string (stored verbatim in the result).
            confidence: OCR confidence score in [0, 1].

        Returns:
            Parsed message or None if it should be discarded.
        """
        if confidence < self._min_confidence:
            logger.debug("Confidence {:.2f} below threshold — discarding", confidence)
            return None

        text = text.strip()
        if not text:
            return None

        msg = self._match(text, raw_ocr)
        if msg is None:
            # Treat unrecognised lines as system messages
            msg = ParsedMessage(
                chat_type="system", author=None, message=text, raw_ocr=raw_ocr
            )

        # Filter by ignored players
        if msg.author and msg.author.lower() in self._ignore_players:
            logger.debug("Ignoring message from player {!r}", msg.author)
            return None

        # Filter by keywords
        if self._keyword_filters:
            lower_msg = msg.message.lower()
            if not any(kw in lower_msg for kw in self._keyword_filters):
                return None

        # Deduplication
        if self._buffer.is_duplicate(msg):
            return None

        self._buffer.add(msg)
        return msg

    def parse_lines(
        self, raw_ocr: str, confidence: float = 1.0
    ) -> List[ParsedMessage]:
        """Parse every non-empty line from a multi-line OCR result.

        Args:
            raw_ocr: Full multi-line string from the OCR engine.
            confidence: OCR confidence to apply to every line.

        Returns:
            List of accepted ParsedMessage objects (may be empty).
        """
        messages: List[ParsedMessage] = []
        for line in raw_ocr.splitlines():
            line = line.strip()
            if not line:
                continue
            msg = self.parse(line, raw_ocr=line, confidence=confidence)
            if msg is not None:
                messages.append(msg)
        return messages

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _match(text: str, raw_ocr: str) -> Optional[ParsedMessage]:
        """Try each pattern against *text*.

        Args:
            text: Single chat line.
            raw_ocr: Original OCR string.

        Returns:
            ParsedMessage on a successful match or None.
        """
        for chat_type, pattern in _PATTERNS:
            m = pattern.match(text)
            if m:
                groups = m.groupdict()
                return ParsedMessage(
                    chat_type=chat_type,
                    author=groups.get("author"),
                    message=groups.get("message", text),
                    raw_ocr=raw_ocr,
                )
        return None
