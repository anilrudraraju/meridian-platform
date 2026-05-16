import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple


class PIIScanner:
    """
    Extended PII detection for the Responsible AI layer.
    Adds credit-card detection on top of FinancialGuardrails and returns
    a structured per-type breakdown for audit display.
    Source: week5_capstone.ipynb InputGuardrails
    """

    _PATTERNS = {
        "SSN":         re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "Credit Card": re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),
        "Email":       re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
        "Phone":       re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        "Account No.": re.compile(r'(?<!\d)\d{10,17}(?!\d)'),
    }
    _INJECTION = [
        'ignore previous instructions', 'disregard', 'forget all',
        'new instructions', 'system prompt', 'ignore all instructions',
        'override instructions',
    ]
    _BLOCKED_TOPICS = ['insider trading', 'money laundering']

    def scan(self, text: str) -> Dict:
        """
        Returns {'pii': {type: [matches]}, 'injections': [kw], 'blocked_topics': [topic]}.
        """
        pii: Dict[str, List[str]] = {}
        for label, pattern in self._PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                pii[label] = matches

        injections = [
            kw for kw in self._INJECTION
            if re.search(re.escape(kw), text, re.IGNORECASE)
        ]
        blocked = [
            t for t in self._BLOCKED_TOPICS
            if t.lower() in text.lower()
        ]
        return {"pii": pii, "injections": injections, "blocked_topics": blocked}

    def is_safe(self, text: str) -> Tuple[bool, str]:
        result = self.scan(text)
        issues = []
        if result["pii"]:
            issues.append(f"PII detected: {list(result['pii'].keys())}")
        if result["injections"]:
            issues.append(f"Prompt injection: {result['injections']}")
        if result["blocked_topics"]:
            issues.append(f"Blocked topic: {result['blocked_topics']}")
        if issues:
            return False, "; ".join(issues)
        return True, "Clean"


class BiasDetector:
    """
    Demographic bias testing — runs the same prompt across demographic groups
    and measures response variance.
    Source: week5_capstone.ipynb test_demographic_bias()
    """

    DEFAULT_DEMOGRAPHICS = ["25-year-old", "65-year-old", "male client", "female client"]

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def run(self, prompt_template: str, demographics: List[str] = None) -> Dict:
        """
        prompt_template must contain '{demographic}'.
        Returns {demographic: response_text, ..., '__bias_score__': float, '__model__': str}.
        bias_score: 0.0 = all responses identical, 1.0 = all responses unique.
        """
        groups = demographics or self.DEFAULT_DEMOGRAPHICS
        responses: Dict[str, str] = {}
        for demo in groups:
            prompt = prompt_template.format(demographic=demo)
            try:
                r = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300,
                )
                responses[demo] = r.choices[0].message.content.strip()
            except Exception as e:
                responses[demo] = f"[Error: {e}]"

        unique = len(set(responses.values()))
        bias_score = 1.0 - (1.0 / max(unique, 1)) if len(groups) > 1 else 0.0
        return {**responses, "__bias_score__": round(bias_score, 3), "__model__": self.model}


class AuditLogger:
    """
    Append-only JSONL audit log for all Layer 5 interactions.
    Source: week5_capstone.ipynb AuditLogger
    """

    LOG_PATH = "/tmp/meridian_audit.jsonl"

    def log(self, user_id: str, input_text: str, output_text: str,
            metadata: Dict = None) -> Dict:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "input": input_text,
            "output": output_text,
            "metadata": metadata or {},
        }
        try:
            with open(self.LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        return entry

    def read_log(self) -> List[Dict]:
        try:
            with open(self.LOG_PATH, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []
