from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class PromptResult:
    """week1_capstone.ipynb"""
    prompt: str
    response: str
    model: str
    tokens_used: int
    cost_estimate: float
    timestamp: str
    technique: str

    def __repr__(self):
        return f"PromptResult(technique={self.technique}, tokens={self.tokens_used}, cost=${self.cost_estimate:.4f})"


@dataclass
class GuardrailResult:
    """week1_capstone.ipynb"""
    passed: bool
    message: str
    violations: List[str]
    modified_content: Optional[str] = None


@dataclass
class SearchResult:
    """week3_capstone.ipynb"""
    content: str
    source: str
    relevance_score: float
    metadata: Dict


@dataclass
class RAGResponse:
    """week3_capstone.ipynb"""
    question: str
    answer: str
    sources: List[SearchResult]
    confidence: str
