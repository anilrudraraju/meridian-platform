import os
import streamlit as st
from datetime import datetime
from typing import Optional

from core.dataclasses import PromptResult
from core.cost import check_budget, log_call


class FinancialPromptEngine:
    """
    Prompt Engineering Engine for Financial Advisory
    Zero-shot · Few-shot · Chain-of-Thought · Role-based · ReAct
    Source: week1_capstone.ipynb
    """

    def __init__(self, model="gpt-4o"):
        import openai
        self.model = model
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.token_costs = {
            "gpt-5":         {"prompt": 0.000625/1000, "completion": 0.005/1000},
            "o3":            {"prompt": 0.002/1000,    "completion": 0.008/1000},
            "o3-mini":       {"prompt": 0.00055/1000,  "completion": 0.0022/1000},
            "gpt-4o":        {"prompt": 0.0025/1000,   "completion": 0.010/1000},
            "gpt-4":         {"prompt": 0.030/1000,    "completion": 0.060/1000},
            "gpt-4o-mini":   {"prompt": 0.00015/1000,  "completion": 0.0006/1000},
            "gpt-4.1-nano":  {"prompt": 0.0001/1000,   "completion": 0.0004/1000},
            "gpt-3.5-turbo": {"prompt": 0.0005/1000,   "completion": 0.001/1000},
        }

    def execute_prompt(self, prompt: str, temperature: float = 0.7,
                       max_tokens: int = 1000, technique: str = "zero-shot") -> Optional[PromptResult]:
        under_budget, spent, cap = check_budget()
        if not under_budget:
            st.error(f"❌ Daily budget cap of ${cap:.2f} reached (spent ${spent:.4f}). LLM call blocked.")
            return None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            p_tok = response.usage.prompt_tokens
            c_tok = response.usage.completion_tokens
            if self.model not in self.token_costs:
                st.warning(f"⚠️ No cost data for model '{self.model}' — using gpt-4o pricing as estimate.")
            costs = self.token_costs.get(self.model, self.token_costs["gpt-4o"])
            cost = p_tok * costs["prompt"] + c_tok * costs["completion"]
            log_call(
                model=self.model,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                cost_usd=cost,
                technique=technique,
                caller="FinancialPromptEngine",
            )
            return PromptResult(
                prompt=prompt, response=content, model=self.model,
                tokens_used=tokens_used, cost_estimate=cost,
                timestamp=datetime.now().isoformat(), technique=technique
            )
        except Exception as e:
            st.error(f"❌ execute_prompt error: {e}")
            return None

    def portfolio_risk_analysis(self, portfolio_data: str) -> Optional[PromptResult]:
        """Zero-shot — week1_capstone.ipynb TEMPLATE 1"""
        prompt = f"""You are an expert financial advisor with 20 years of experience.

Analyze the following portfolio and identify the top 3 risks:

Portfolio Holdings:
{portfolio_data}

Provide your analysis in this format:
Risk 1: [Description]
- Why it matters: [Explanation]
- Mitigation: [Strategy]

Risk 2: [Description]
- Why it matters: [Explanation]
- Mitigation: [Strategy]

Risk 3: [Description]
- Why it matters: [Explanation]
- Mitigation: [Strategy]
"""
        return self.execute_prompt(prompt, technique="zero-shot")

    def portfolio_report_fewshot(self, portfolio_data: str) -> Optional[PromptResult]:
        """Few-shot — week1_capstone.ipynb TEMPLATE 2"""
        prompt = f"""You are a wealth management advisor. Generate a comprehensive portfolio report.

Here are examples of well-formatted reports:

Example 1:
Portfolio: 60% Large-Cap Stocks, 30% Bonds, 10% Cash
Report: "This balanced portfolio demonstrates a moderate risk profile appropriate for investors with a 10-15 year time horizon."

Example 2:
Portfolio: 80% Technology Stocks, 15% Growth Stocks, 5% Cash
Report: "This aggressive growth portfolio shows high concentration in the technology sector (80%), creating significant sector-specific risk."

Example 3:
Portfolio: 40% Dividend Stocks, 35% Bonds, 15% REITs, 10% Cash
Report: "This income-focused portfolio is well-suited for investors prioritizing stable cash flow."

Now, generate a similar detailed report for this portfolio:
Portfolio: {portfolio_data}

Report:"""
        return self.execute_prompt(prompt, technique="few-shot")

    def tax_loss_harvesting_cot(self, holdings_data: str) -> Optional[PromptResult]:
        """Chain-of-Thought — week1_capstone.ipynb TEMPLATE 3"""
        prompt = f"""You are a tax optimization specialist.

Analyze these holdings for tax-loss harvesting opportunities. Think step by step:

Holdings:
{holdings_data}

Step 1: Identify positions with unrealized losses
Step 2: Calculate tax benefit (assume 30% tax rate)
Step 3: Suggest replacement securities (avoid wash sale rules)
Step 4: Prioritize opportunities by tax savings
Step 5: Final recommendation with clear action items

Work through each step methodically."""
        return self.execute_prompt(prompt, temperature=0.3, technique="chain-of-thought")

    def client_communication(self, situation: str, client_type: str = "conservative") -> Optional[PromptResult]:
        """Role-based — week1_capstone.ipynb TEMPLATE 4"""
        roles = {
            "conservative": "You are a trusted financial advisor speaking to a risk-averse client who values stability.",
            "aggressive": "You are a financial advisor working with a sophisticated client comfortable with volatility.",
            "balanced": "You are a financial advisor serving a client seeking reasonable growth while managing risk."
        }
        prompt = f"""{roles.get(client_type, roles['balanced'])}

Situation:
{situation}

Draft a professional client email (under 200 words) with appropriate disclaimers.

Email:"""
        return self.execute_prompt(prompt, temperature=0.8, technique="role-based")

    def market_commentary_react(self, market_event: str) -> Optional[PromptResult]:
        """ReAct — week1_capstone.ipynb TEMPLATE 5"""
        prompt = f"""You are a financial analyst. Use the ReAct framework to analyze this market event.

Market Event: {market_event}

Thought 1: What information do I need?
Action 1: [List data to gather]
Observation 1: [State what it shows]

Thought 2: How does this impact asset classes?
Action 2: [Reason through implications]
Observation 2: [State expected impacts]

Thought 3: What should investors consider?
Action 3: [Develop recommendations]

Final Analysis: [Synthesize for clients]"""
        return self.execute_prompt(prompt, max_tokens=1500, technique="react")
