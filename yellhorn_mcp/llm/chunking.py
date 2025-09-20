"""Chunking strategies for splitting large prompts safely by token limits."""

from typing import List

from yellhorn_mcp.utils.token_utils import TokenCounter


class ChunkingStrategy:
    @staticmethod
    def _find_split_point(text: str, max_length: int) -> int:
        para_break = text.rfind("\n", 0, max_length)
        if para_break > 0:
            return para_break + 2

        sentence_break = max(
            text.rfind(". ", 0, max_length),
            text.rfind("! ", 0, max_length),
            text.rfind("? ", 0, max_length),
            text.rfind("\n", 0, max_length),
        )
        if sentence_break > 0:
            return sentence_break + 1

        space_break = text.rfind(" ", 0, max_length)
        if space_break > 0:
            return space_break

        return max_length

    @staticmethod
    def split_by_sentences(
        text: str,
        max_tokens: int,
        token_counter: TokenCounter,
        model: str,
        overlap_ratio: float = 0.1,
        safety_margin_tokens: int = 50,
    ) -> List[str]:
        if not text.strip():
            return []

        target_tokens = max_tokens - safety_margin_tokens
        chunks: list[str] = []
        remaining_text = text
        overlap_tokens = int(max_tokens * overlap_ratio)

        while remaining_text:
            estimated_tokens = token_counter.count_tokens(remaining_text, model)
            # Fallback heuristic by characters if token estimator says it fits but content is large
            must_split_by_chars = len(remaining_text) > target_tokens * 4
            if estimated_tokens > target_tokens or must_split_by_chars:
                low, high = 0, len(remaining_text)
                best_split = len(remaining_text)
                while low <= high:
                    mid = (low + high) // 2
                    chunk = remaining_text[:mid]
                    tokens = token_counter.count_tokens(chunk, model)
                    if tokens <= target_tokens:
                        best_split = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                # If we only split by char heuristic, tighten split to roughly target size
                if must_split_by_chars and best_split == len(remaining_text):
                    best_split = min(len(remaining_text), max(1, target_tokens * 4))
                split_pos = (
                    ChunkingStrategy._find_split_point(remaining_text[:best_split], best_split)
                    if best_split < len(remaining_text)
                    else best_split
                )
                if split_pos == 0:
                    split_pos = best_split
            else:
                split_pos = len(remaining_text)

            chunk = remaining_text[:split_pos].strip()
            remaining_text = remaining_text[split_pos:].strip()
            if not chunk:
                break
            chunks.append(chunk)

            if remaining_text and overlap_tokens > 0:
                next_sentence_start = 0
                for i, c in enumerate(remaining_text):
                    if c in ".!?":
                        next_sentence_start = i + 1
                        if (
                            next_sentence_start < len(remaining_text)
                            and remaining_text[next_sentence_start] == " "
                        ):
                            next_sentence_start += 1
                        break
                if 0 < next_sentence_start < len(remaining_text):
                    overlap_text = remaining_text[:next_sentence_start]
                    remaining_text = overlap_text + remaining_text[next_sentence_start:]

        return chunks

    @staticmethod
    def split_by_paragraphs(
        text: str,
        max_tokens: int,
        token_counter: TokenCounter,
        model: str,
        overlap_ratio: float = 0.1,
        safety_margin_tokens: int = 50,
    ) -> List[str]:
        if not text.strip():
            return []

        target_tokens = max_tokens - safety_margin_tokens
        paragraphs = [p for p in text.split("\n") if p.strip()]
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = token_counter.count_tokens(para, model)
            if para_tokens > target_tokens:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk, current_tokens = [], 0
                chunks.extend(
                    ChunkingStrategy.split_by_sentences(
                        para, max_tokens, token_counter, model, overlap_ratio, safety_margin_tokens
                    )
                )
            elif current_tokens + para_tokens > target_tokens and current_chunk:
                chunks.append("\n".join(current_chunk))
                if overlap_ratio > 0 and chunks:
                    overlap_tokens = int(max_tokens * overlap_ratio)
                    overlap_text = "\n".join(current_chunk)[-overlap_tokens * 4 :]
                    current_chunk = [overlap_text, para]
                    current_tokens = token_counter.count_tokens("\n".join(current_chunk), model)
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens + 2

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks
