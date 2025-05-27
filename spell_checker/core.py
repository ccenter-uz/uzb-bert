import json
import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional, Union
from transformers import pipeline, Pipeline
from spell_checker.utils import normalize_token, levenshtein
from spell_checker.config import settings
from difflib import get_close_matches

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

class Suggestion:
    def __init__(self, candidate: str, score: float):
        self.candidate = candidate
        self.score = score

    def as_dict(self):
        return {
            "candidate": self.candidate,
            "score": round(self.score, 4)
        }


class Flag:
    def __init__(self, token: str, char_start: int, char_end: int, suggestions: List[Suggestion]):
        self.token = token
        self.char_start = char_start
        self.char_end = char_end
        self.suggestions = suggestions

    def as_dict(self):
        return {
            "token": self.token,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "suggestions": [s.as_dict() for s in self.suggestions]
        }


class SpellChecker:
    def __init__(self, dict_path: Optional[str] = None):
        self._dict_path = dict_path or settings.DICTIONARY_PATH
        self._model: Optional[Pipeline] = None
        self._dictionary: Optional[set] = None
        self._mask_token: Optional[str] = None
        
    @property
    def model(self) -> Pipeline:
        """Lazy initialization of the model"""
        if self._model is None:
            try:
                logger.info(f"Loading model: {settings.MODEL_NAME}")
                self._model = pipeline("fill-mask", model=settings.MODEL_NAME)
                self._mask_token = self._model.tokenizer.mask_token
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        return self._model

    @property
    def dictionary(self) -> set:
        """Lazy loading of dictionary"""
        if self._dictionary is None:
            try:
                logger.info(f"Loading dictionary from: {self._dict_path}")
                with open(self._dict_path, encoding="utf-8") as f:
                    self._dictionary = set(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load dictionary: {e}")
                raise
        return self._dictionary

    @property
    def mask_token(self) -> str:
        if self._mask_token is None:
            _ = self.model  # Ensure model is loaded
        return self._mask_token

    def is_valid_word(self, token: str) -> bool:
        return token in self.dictionary

    @lru_cache(maxsize=1024)
    def fuzzy_match_from_dict(self, word: str, k: int) -> List[Suggestion]:
        matches = get_close_matches(word, self.dictionary, n=k, cutoff=0.8)
        return [Suggestion(candidate=m, score=0.001) for m in matches]

    def suggest_for_word(self, sentence: str, word: str, k: int) -> List[Suggestion]:
        try:
            match = re.search(rf"\b{re.escape(word)}\b", sentence)
            if not match:
                logger.warning(f"Word '{word}' not found in sentence")
                return self.fuzzy_match_from_dict(word, k)

            start, end = match.start(), match.end()
            left_context = sentence[max(0, start - 40):start].strip()
            right_context = sentence[end:end + 40].strip()
            masked_sentence = f"{left_context} {self.mask_token} {right_context}"

            if self.mask_token not in masked_sentence:
                logger.warning(f"Masking failed for word: {word}")
                return self.fuzzy_match_from_dict(word, k)

            results = self.model(masked_sentence, top_k=settings.MODEL_TOP_K)
            
            suggestions = []
            seen = set()
            
            for r in results:
                if r["score"] < 0.001:
                    continue

                candidate = normalize_token(r["token_str"])
                dist = levenshtein(candidate, word)

                if (
                    candidate not in seen and
                    candidate in self.dictionary and
                    0 < dist <= 3
                ):
                    final_score = r["score"] * (1 - (dist / 3))
                    suggestions.append(Suggestion(candidate, final_score))
                    seen.add(candidate)

                if len(suggestions) >= k:
                    break

            if not suggestions:
                logger.info(f"No BERT suggestions found for '{word}', using fuzzy matching")
                suggestions = self.fuzzy_match_from_dict(word, k)

            return suggestions

        except Exception as e:
            logger.error(f"Error suggesting words for '{word}': {e}")
            return self.fuzzy_match_from_dict(word, k)

    def check_batch(self, texts: List[str], max_suggestions: int = 3) -> List[Dict]:
        """Process multiple texts in batch"""
        return [self.check(text, max_suggestions) for text in texts]

    def check(self, text: str, max_suggestions: int = 3) -> Dict:
        try:
            words = text.split()
            char_pos = 0
            flags = []

            for word in words:
                clean = normalize_token(word)
                start = text.find(word, char_pos)
                end = start + len(word)
                char_pos = end

                if not clean or self.is_valid_word(clean):
                    continue

                suggestions = self.suggest_for_word(text, clean, max_suggestions)
                flags.append(Flag(token=clean, char_start=start, char_end=end, suggestions=suggestions))

            return {
                "original_text": text,
                "flags": [f.as_dict() for f in flags]
            }
        except Exception as e:
            logger.error(f"Error checking text: {e}")
            return {
                "original_text": text,
                "flags": [],
                "error": str(e)
            }
