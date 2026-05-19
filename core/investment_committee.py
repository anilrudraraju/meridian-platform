"""
core/investment_committee.py — Layer 9: Agent Communication & Collaborative Decision-Making
Source: week9_capstone
Framework: OpenAI direct LLM calls with structured MessageBus communication protocol
3-round debate: Initial Positions → Challenges → Final Vote
"""
import os
from typing import List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


# ── Message infrastructure ─────────────────────────────────────────────────────

@dataclass
class Message:
    sender: str
    recipient: str      # agent role or 'all'
    content: str
    timestamp: str
    message_type: str   # 'proposal' | 'challenge' | 'vote'
    round_num: int


class MessageBus:
    """Central message bus — append-only log of all inter-agent communication."""

    def __init__(self):
        self.messages: List[Message] = []

    def send(self, message: Message):
        self.messages.append(message)

    def get_by_round(self, round_num: int) -> List[Message]:
        return [m for m in self.messages if m.round_num == round_num]

    def context_for(self, agent_role: str, before_round: int) -> str:
        """Return all messages from other agents in rounds before `before_round`."""
        relevant = [
            m for m in self.messages
            if m.round_num < before_round and m.sender != agent_role
        ]
        return "\n\n".join(
            f"[{m.sender}] ({m.message_type}): {m.content}" for m in relevant
        )

    def full_transcript(self) -> str:
        return "\n\n".join(
            f"[Round {m.round_num}] {m.sender} ({m.message_type}): {m.content}"
            for m in self.messages
        )


# ── Agent definitions ──────────────────────────────────────────────────────────

_AGENTS = [
    {
        "role": "Growth Investment Specialist",
        "icon": "📈",
        "system": (
            "You are a growth-focused investor at Meridian Wealth Partners with 15 years "
            "of experience in technology and innovation sectors. You believe in identifying "
            "disruptive companies early and holding through volatility. You prioritise revenue "
            "growth, TAM expansion, and market leadership over traditional valuation metrics. "
            "Be direct, opinionated, and keep responses under 200 words."
        ),
    },
    {
        "role": "Value Investment Specialist",
        "icon": "💰",
        "system": (
            "You are a value investor at Meridian Wealth Partners following the Buffett/Graham "
            "tradition. You focus on companies trading below intrinsic value with solid balance "
            "sheets, consistent earnings, and durable competitive moats. You are sceptical of "
            "high-multiple momentum plays and always ask what you are paying for. "
            "Be direct, data-grounded, and keep responses under 200 words."
        ),
    },
    {
        "role": "Chief Risk Officer",
        "icon": "🛡️",
        "system": (
            "You are the quantitative risk manager at Meridian Wealth Partners. You assess "
            "concentration risk, correlation, volatility, drawdown scenarios, and liquidity. "
            "Your job is to be the voice of caution — surface what could go wrong and "
            "quantify the potential impact. Be direct, specific, and keep responses under 200 words."
        ),
    },
]

_ROUND1_PROMPT = """Investment Proposal: {proposal}

Provide your initial position:
1. Your stance given your investment philosophy (1 sentence)
2. Top 2 benefits you see
3. Top 2 concerns you have
4. Preliminary recommendation: APPROVE, REJECT, or MODIFY

Be specific and stay under 200 words."""

_ROUND2_PROMPT = """Investment Proposal: {proposal}

Other committee members' initial positions:
{context}

Respond to the discussion:
1. Challenge the weakest argument from another member (name them explicitly)
2. Acknowledge one valid point from another member (name them)
3. State whether your position has shifted and why

Stay under 200 words. Be direct."""

_ROUND3_PROMPT = """Investment Proposal: {proposal}

Full committee discussion so far:
{context}

Cast your final vote. Use exactly this format:

VOTE: [APPROVE / REJECT / MODIFY]
JUSTIFICATION: [2–3 sentences citing specific points from the debate]"""


# ── Committee class ────────────────────────────────────────────────────────────

class InvestmentCommittee:
    """
    Three-agent investment committee that debates proposals through a structured
    3-round protocol:
      Round 1 — Initial Positions  (message_type: 'proposal')
      Round 2 — Challenges         (message_type: 'challenge')
      Round 3 — Final Vote         (message_type: 'vote')

    Outcome: majority vote → APPROVED / REJECTED / NO CONSENSUS
    """

    def __init__(self, model: str = "gpt-4o"):
        import openai
        self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def _call(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _parse_vote(text: str) -> str:
        upper = text.upper()
        if "APPROVE" in upper:
            return "APPROVE"
        if "REJECT" in upper:
            return "REJECT"
        if "MODIFY" in upper:
            return "MODIFY"
        return "ABSTAIN"

    def run(
        self,
        proposal: str,
        on_message: Optional[Callable] = None,
    ) -> dict:
        """
        Run the full 3-round debate.

        on_message(round_num, agent_role, icon, message_type, content)
        is called after each agent responds — use it to stream live UI updates.

        Returns:
          {proposal, bus, votes, tally, outcome, agent_outputs, message_count}
        """
        bus = MessageBus()
        agent_outputs = []

        # ── Round 1: Initial Positions ────────────────────────────────────────
        for agent in _AGENTS:
            content = self._call(
                agent["system"],
                _ROUND1_PROMPT.format(proposal=proposal),
            )
            bus.send(Message(
                sender=agent["role"], recipient="all", content=content,
                timestamp=datetime.now().isoformat(),
                message_type="proposal", round_num=1,
            ))
            agent_outputs.append({"round": 1, "agent": agent["role"], "icon": agent["icon"], "content": content})
            if on_message:
                on_message(1, agent["role"], agent["icon"], "proposal", content)

        # ── Round 2: Challenges ───────────────────────────────────────────────
        for agent in _AGENTS:
            context = bus.context_for(agent["role"], before_round=2)
            content = self._call(
                agent["system"],
                _ROUND2_PROMPT.format(proposal=proposal, context=context),
            )
            bus.send(Message(
                sender=agent["role"], recipient="all", content=content,
                timestamp=datetime.now().isoformat(),
                message_type="challenge", round_num=2,
            ))
            agent_outputs.append({"round": 2, "agent": agent["role"], "icon": agent["icon"], "content": content})
            if on_message:
                on_message(2, agent["role"], agent["icon"], "challenge", content)

        # ── Round 3: Final Vote ───────────────────────────────────────────────
        votes = {}
        for agent in _AGENTS:
            content = self._call(
                agent["system"],
                _ROUND3_PROMPT.format(proposal=proposal, context=bus.full_transcript()),
            )
            vote = self._parse_vote(content)
            votes[agent["role"]] = {"vote": vote, "justification": content, "icon": agent["icon"]}
            bus.send(Message(
                sender=agent["role"], recipient="all", content=content,
                timestamp=datetime.now().isoformat(),
                message_type="vote", round_num=3,
            ))
            agent_outputs.append({
                "round": 3, "agent": agent["role"], "icon": agent["icon"],
                "content": content, "vote": vote,
            })
            if on_message:
                on_message(3, agent["role"], agent["icon"], "vote", content)

        # ── Tally ─────────────────────────────────────────────────────────────
        tally: dict = {}
        for v in votes.values():
            tally[v["vote"]] = tally.get(v["vote"], 0) + 1

        n = len(_AGENTS)
        if tally.get("APPROVE", 0) > n / 2:
            outcome = "APPROVED"
        elif tally.get("REJECT", 0) > n / 2:
            outcome = "REJECTED"
        else:
            outcome = "NO CONSENSUS — MODIFICATION REQUIRED"

        return {
            "proposal": proposal,
            "bus": bus,
            "votes": votes,
            "tally": tally,
            "outcome": outcome,
            "agent_outputs": agent_outputs,
            "message_count": len(bus.messages),
        }
