"""Tests for chat_parser.py"""

import time

import pytest

from chat_parser import ChatParser, ParsedMessage, _MessageBuffer


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

class TestChatParserPatterns:
    def setup_method(self):
        self.parser = ChatParser(min_confidence=0.0)

    def _parse(self, text: str) -> ParsedMessage:
        msg = self.parser.parse(text, raw_ocr=text, confidence=1.0)
        assert msg is not None, f"Expected a message but got None for: {text!r}"
        return msg

    def test_public_chat(self):
        msg = self._parse("[Zezima] Nice bank")
        assert msg.chat_type == "public"
        assert msg.author == "Zezima"
        assert msg.message == "Nice bank"

    def test_clan_chat(self):
        msg = self._parse("[Clan] [Leader] Great job everyone")
        assert msg.chat_type == "clan"
        assert msg.author == "Leader"
        assert "Great job" in msg.message

    def test_guest_clan_chat(self):
        msg = self._parse("[Guest Clan] [Visitor] Hello!")
        assert msg.chat_type == "guest_clan"
        assert msg.author == "Visitor"

    def test_private_inbound(self):
        msg = self._parse("From PlayerOne: Are you free?")
        assert msg.chat_type == "private_in"
        assert msg.author == "PlayerOne"
        assert msg.message == "Are you free?"

    def test_private_outbound(self):
        msg = self._parse("To PlayerTwo: On my way")
        assert msg.chat_type == "private_out"
        assert msg.author == "PlayerTwo"
        assert msg.message == "On my way"

    def test_trade(self):
        msg = self._parse("Trade with HighValuePlayer")
        assert msg.chat_type == "trade"
        assert msg.author == "HighValuePlayer"

    def test_system_fallback(self):
        msg = self._parse("Welcome to Old School RuneScape.")
        assert msg.chat_type == "system"
        assert msg.author is None

    def test_empty_text_returns_none(self):
        result = self.parser.parse("   ", raw_ocr="", confidence=1.0)
        assert result is None

    def test_low_confidence_filtered(self):
        parser = ChatParser(min_confidence=0.9)
        result = parser.parse("[Player] Hello", raw_ocr="[Player] Hello", confidence=0.5)
        assert result is None


# ---------------------------------------------------------------------------
# Keyword and ignore filters
# ---------------------------------------------------------------------------

class TestChatParserFilters:
    def test_keyword_filter_pass(self):
        parser = ChatParser(keyword_filters=["trade"], min_confidence=0.0)
        msg = parser.parse("[Bob] WTS rune scimitar trade", raw_ocr="", confidence=1.0)
        assert msg is not None

    def test_keyword_filter_block(self):
        parser = ChatParser(keyword_filters=["partyhat"], min_confidence=0.0)
        result = parser.parse("[Bob] Hello there", raw_ocr="", confidence=1.0)
        assert result is None

    def test_ignore_player(self):
        parser = ChatParser(ignore_players=["SpamBot"], min_confidence=0.0)
        result = parser.parse("[SpamBot] Buy gold now!", raw_ocr="", confidence=1.0)
        assert result is None

    def test_ignore_player_case_insensitive(self):
        parser = ChatParser(ignore_players=["spambot"], min_confidence=0.0)
        result = parser.parse("[SpamBot] Buy gold now!", raw_ocr="", confidence=1.0)
        assert result is None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_message_blocked(self):
        parser = ChatParser(min_confidence=0.0)
        text = "[Player] Hello world"
        first = parser.parse(text, raw_ocr=text, confidence=1.0)
        assert first is not None
        # Identical message should be deduplicated
        second = parser.parse(text, raw_ocr=text, confidence=1.0)
        assert second is None

    def test_similar_message_blocked(self):
        parser = ChatParser(min_confidence=0.0)
        text1 = "[Player] Hello world, how are you?"
        text2 = "[Player] Hello world, how are you?"  # exact same
        parser.parse(text1, raw_ocr=text1, confidence=1.0)
        result = parser.parse(text2, raw_ocr=text2, confidence=1.0)
        assert result is None

    def test_different_message_passes(self):
        parser = ChatParser(min_confidence=0.0)
        parser.parse("[Player] Hello", raw_ocr="[Player] Hello", confidence=1.0)
        result = parser.parse("[Player] Goodbye", raw_ocr="[Player] Goodbye", confidence=1.0)
        assert result is not None


# ---------------------------------------------------------------------------
# parse_lines
# ---------------------------------------------------------------------------

class TestParseLines:
    def test_multi_line(self):
        parser = ChatParser(min_confidence=0.0)
        ocr_text = "[Alice] Hey\n[Bob] Wassup\n"
        messages = parser.parse_lines(ocr_text, confidence=1.0)
        assert len(messages) == 2
        authors = {m.author for m in messages}
        assert "Alice" in authors
        assert "Bob" in authors

    def test_empty_lines_skipped(self):
        parser = ChatParser(min_confidence=0.0)
        messages = parser.parse_lines("\n\n\n", confidence=1.0)
        assert messages == []


# ---------------------------------------------------------------------------
# MessageBuffer TTL
# ---------------------------------------------------------------------------

class TestMessageBuffer:
    def test_expired_entries_evicted(self):
        buf = _MessageBuffer(max_size=10, ttl=0.01)
        msg = ParsedMessage(chat_type="public", author="A", message="hi", raw_ocr="hi")
        buf.add(msg)
        time.sleep(0.05)
        # After TTL, should not be considered a duplicate
        assert not buf.is_duplicate(msg)
