"""
core/rebalancing_workflow.py — Layer 8: Stateful Portfolio Rebalancing with LangGraph
Source: week8_capstone
Framework: LangGraph StateGraph with conditional edges + human-in-the-loop interrupt
"""
from typing import TypedDict, List, Annotated, Optional, Callable
import operator


# ── State ─────────────────────────────────────────────────────────────────────

class RebalancingState(TypedDict):
    portfolio: dict           # {ticker: dollar_value}
    target_allocation: dict   # {ticker: decimal_weight}
    current_allocation: dict  # computed
    drift: float
    total_value: float
    trades: List[dict]
    total_trade_value: float
    tax_impact: float
    approval_threshold: float # configurable, default $1M
    requires_approval: bool
    manual_approved: Optional[bool]  # set by UI before phase-2 resume
    approved: bool
    status: str
    messages: Annotated[List[str], operator.add]  # reducer: append, never replace


# ── Nodes ─────────────────────────────────────────────────────────────────────

def check_drift_node(state: RebalancingState) -> dict:
    total_value = sum(state["portfolio"].values())
    current_alloc = {t: v / total_value for t, v in state["portfolio"].items()}
    drift = sum(
        abs(current_alloc.get(t, 0) - w)
        for t, w in state["target_allocation"].items()
    )
    return {
        "total_value": total_value,
        "current_allocation": current_alloc,
        "drift": drift,
        "messages": [
            f"Drift analysis: {drift:.2%} from target "
            f"(portfolio value ${total_value:,.0f})"
        ],
    }


def generate_trades_node(state: RebalancingState) -> dict:
    trades = []
    total = state["total_value"]
    for ticker, target_pct in state["target_allocation"].items():
        current_value = state["portfolio"].get(ticker, 0)
        target_value = total * target_pct
        trade_value = target_value - current_value
        if abs(trade_value) > 1_000:
            trades.append({
                "ticker": ticker,
                "action": "buy" if trade_value > 0 else "sell",
                "value": abs(trade_value),
                "current_value": current_value,
                "target_value": target_value,
            })
    total_trade = sum(t["value"] for t in trades)
    return {
        "trades": trades,
        "total_trade_value": total_trade,
        "messages": [f"Generated {len(trades)} trades totalling ${total_trade:,.0f}"],
    }


def optimize_tax_node(state: RebalancingState) -> dict:
    sell_trades = [t for t in state["trades"] if t["action"] == "sell"]
    sell_total = sum(t["value"] for t in sell_trades)
    tax = sell_total * 0.20  # simplified 20% capital gains
    return {
        "tax_impact": tax,
        "messages": [
            f"Tax estimate: ${tax:,.0f} "
            f"(20% rate on ${sell_total:,.0f} in sell trades)"
        ],
    }


def check_approval_node(state: RebalancingState) -> dict:
    threshold = state.get("approval_threshold", 1_000_000)
    requires = state["total_trade_value"] > threshold
    msg = (
        f"Approval required — ${state['total_trade_value']:,.0f} "
        f"exceeds ${threshold:,.0f} threshold"
        if requires
        else f"No approval required — ${state['total_trade_value']:,.0f} "
             f"below ${threshold:,.0f} threshold"
    )
    return {"requires_approval": requires, "messages": [msg]}


def human_approval_node(state: RebalancingState) -> dict:
    # Runs only in phase 2 after the UI has injected manual_approved
    approved = bool(state.get("manual_approved"))
    return {
        "approved": approved,
        "messages": [f"Human decision: {'APPROVED ✅' if approved else 'REJECTED ❌'}"],
    }


def execute_trades_node(state: RebalancingState) -> dict:
    lines = [
        f"{t['action'].upper()} {t['ticker']} ${t['value']:,.0f}"
        for t in state["trades"]
    ]
    return {
        "status": "completed",
        "messages": [f"Executed {len(state['trades'])} trades: " + " | ".join(lines)],
    }


def no_rebalance_node(state: RebalancingState) -> dict:
    return {
        "status": "no_action_needed",
        "messages": [
            f"Drift {state['drift']:.2%} is within the 5% threshold — "
            "no rebalancing required"
        ],
    }


def rejection_node(state: RebalancingState) -> dict:
    return {
        "status": "rejected",
        "messages": ["Rebalancing rejected by reviewer — no trades executed"],
    }


# ── Workflow class ─────────────────────────────────────────────────────────────

class RebalancingWorkflow:
    """
    LangGraph state machine for autonomous portfolio rebalancing.

    Graph:
      check_drift
        → [drift > 5%] → generate_trades → optimize_tax → check_approval
            → [trade > threshold] → ⏸ HUMAN APPROVAL GATE
                → [approved]  → execute_trades → END
                → [rejected]  → rejection      → END
            → [trade ≤ threshold] → execute_trades → END
        → [drift ≤ 5%] → no_rebalance → END
    """

    def __init__(self):
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        g = StateGraph(RebalancingState)

        g.add_node("check_drift",     check_drift_node)
        g.add_node("generate_trades", generate_trades_node)
        g.add_node("optimize_tax",    optimize_tax_node)
        g.add_node("check_approval",  check_approval_node)
        g.add_node("human_approval",  human_approval_node)
        g.add_node("execute_trades",  execute_trades_node)
        g.add_node("no_rebalance",    no_rebalance_node)
        g.add_node("rejection",       rejection_node)

        g.set_entry_point("check_drift")

        # Drift gate: > 5% → rebalance, else → no action
        g.add_conditional_edges(
            "check_drift",
            lambda s: "rebalance" if s["drift"] > 0.05 else "no_action",
            {"rebalance": "generate_trades", "no_action": "no_rebalance"},
        )

        g.add_edge("generate_trades", "optimize_tax")
        g.add_edge("optimize_tax",    "check_approval")

        # Approval gate
        g.add_conditional_edges(
            "check_approval",
            lambda s: "needs_approval" if s["requires_approval"] else "execute",
            {"needs_approval": "human_approval", "execute": "execute_trades"},
        )

        # Post-approval routing
        g.add_conditional_edges(
            "human_approval",
            lambda s: "execute" if s.get("approved") else "end",
            {"execute": "execute_trades", "end": "rejection"},
        )

        g.add_edge("execute_trades", END)
        g.add_edge("no_rebalance",   END)
        g.add_edge("rejection",      END)

        self._memory = MemorySaver()
        self._app = g.compile(
            checkpointer=self._memory,
            interrupt_before=["human_approval"],  # pause before approval node
        )

    def _initial_state(
        self,
        portfolio: dict,
        target_allocation: dict,
        approval_threshold: float,
    ) -> dict:
        return {
            "portfolio":          portfolio,
            "target_allocation":  target_allocation,
            "current_allocation": {},
            "drift":              0.0,
            "total_value":        0.0,
            "trades":             [],
            "total_trade_value":  0.0,
            "tax_impact":         0.0,
            "approval_threshold": approval_threshold,
            "requires_approval":  False,
            "manual_approved":    None,
            "approved":           False,
            "status":             "pending",
            "messages":           [],
        }

    def analyze(
        self,
        portfolio: dict,
        target_allocation: dict,
        approval_threshold: float = 1_000_000,
        thread_id: str = "default",
        on_node: Optional[Callable] = None,
    ) -> dict:
        """
        Phase 1: run the workflow. If approval is needed the graph pauses
        before human_approval and returns with status='pending_approval'.
        on_node(node_name, updates, accumulated_state) is called after each node.
        """
        config = {"configurable": {"thread_id": thread_id}}
        initial = self._initial_state(portfolio, target_allocation, approval_threshold)
        accumulated = dict(initial)

        for chunk in self._app.stream(initial, config, stream_mode="updates"):
            for node_name, updates in chunk.items():
                for key, value in updates.items():
                    if key == "messages" and isinstance(value, list):
                        accumulated["messages"] = accumulated.get("messages", []) + value
                    else:
                        accumulated[key] = value
                if on_node:
                    on_node(node_name, updates, dict(accumulated))

        # Read persisted final state from checkpointer
        snapshot = self._app.get_state(config)
        result = dict(snapshot.values)
        if result.get("status") == "pending" and result.get("requires_approval"):
            result["status"] = "pending_approval"
        return result

    def complete(self, approved: bool, thread_id: str = "default") -> dict:
        """
        Phase 2: inject the user's approval decision and resume the paused workflow.
        """
        config = {"configurable": {"thread_id": thread_id}}
        self._app.update_state(config, {"manual_approved": approved})
        result = dict(self._app.invoke(None, config))
        return result
