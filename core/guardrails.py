import re
from typing import Tuple, Optional

from core.dataclasses import GuardrailResult, PromptResult
from core.prompts import FinancialPromptEngine


class FinancialGuardrails:
    """Source: week1_capstone.ipynb"""

    def __init__(self):
        # SSN: with or without hyphens (e.g. 123-45-6789 or 123456789)
        self.ssn_pattern     = re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b')
        # Account: 10-17 digits not preceded/followed by another digit (avoids dates/zip codes)
        self.account_pattern = re.compile(r'(?<!\d)\d{10,17}(?!\d)')
        # Phone: standard 10-digit US formats only
        self.phone_pattern   = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.email_pattern   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
        # Injection keywords — checked case-insensitively with word-boundary awareness
        self.injection_keywords = [
            'ignore previous instructions', 'disregard',
            'forget all', 'new instructions', 'system prompt',
            'ignore all instructions', 'override instructions'
        ]
        self.unauthorized_advice = [
            'you should buy', 'i recommend buying', 'guaranteed returns',
            'risk-free investment', 'you should sell', 'definitely invest'
        ]
        self.compliance_disclaimer = (
            "\n\n*This is not financial advice. Consult with a licensed "
            "financial professional before making investment decisions.*"
        )

    def validate_input(self, user_input: str) -> GuardrailResult:
        violations = []
        if self.ssn_pattern.search(user_input):
            violations.append("Social Security Number detected")
        if self.account_pattern.search(user_input):
            violations.append("Account number detected")
        if self.phone_pattern.search(user_input):
            violations.append("Phone number detected")
        for kw in self.injection_keywords:
            if re.search(re.escape(kw), user_input, re.IGNORECASE):
                violations.append(f"Potential prompt injection: '{kw}'")
        if violations:
            return GuardrailResult(passed=False, message="Input validation failed", violations=violations)
        return GuardrailResult(passed=True, message="Input validated successfully", violations=[])

    def validate_output(self, ai_output: str) -> GuardrailResult:
        violations = []
        modified = ai_output
        for phrase in self.unauthorized_advice:
            if re.search(re.escape(phrase), ai_output, re.IGNORECASE):
                violations.append(f"Unauthorized advice: '{phrase}'")
        if self.ssn_pattern.search(ai_output) or self.account_pattern.search(ai_output):
            violations.append("PII detected in output")
        if violations:
            # Block the response entirely rather than masking it with a disclaimer
            modified = (
                "⚠️ Response blocked by compliance guardrails: "
                + "; ".join(violations)
                + self.compliance_disclaimer
            )
            return GuardrailResult(
                passed=False,
                message="Output blocked by compliance guardrails",
                violations=violations,
                modified_content=modified
            )
        # No violations — append disclaimer if not already present
        if "this is not financial advice" not in ai_output.lower():
            modified += self.compliance_disclaimer
        return GuardrailResult(
            passed=True,
            message="Output validated",
            violations=[],
            modified_content=modified
        )

    def safe_execute(self, prompt_engine: FinancialPromptEngine,
                     prompt_function, *args, **kwargs) -> Tuple[bool, Optional[PromptResult]]:
        if args and isinstance(args[0], str):
            check = self.validate_input(args[0])
            if not check.passed:
                return False, None
        result = prompt_function(*args, **kwargs)
        if result is None:
            return False, None
        out_check = self.validate_output(result.response)
        result.response = out_check.modified_content
        return True, result
