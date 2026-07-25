"""
SEC Form 4 (insider transaction) XML parsing.

Form 4s are genuinely structured XML — unlike Form 10/10-K/10-Q, there is
no HTML-table-soup problem here at all, and no LLM involvement belongs in
this path. This module exists to get the mechanical aggregation right:
filtering to actual open-market purchases, not every reported transaction
type, and correctly handling one filing containing multiple transactions.

Confirmed live across all 3 pilots before writing this:
  - The SEC submissions API's primaryDocument for Form 4s always points at
    an XSL-rendered HTML view path ('xslF345XNN/<file>.xml'), not the raw
    XML — DiscoveredFiling.raw_document_url (sec_service.py) strips that
    prefix to get the real machine-parseable file. This module always
    fetches raw_document_url, never sec_url, for that reason.
  - transactionCode values actually seen: P (open-market purchase, INBX),
    S (sale, all 3 pilots), A (grant/award), M (option exercise), F (tax
    withholding), G (gift) — only P is a genuine insider BUY; the others
    must never be counted toward insider_buying_within_3mo, even though
    some (like M, option exercise) increase share count and could be
    superficially mistaken for buying activity.
  - One filing can report MULTIPLE transactions (confirmed live: a single
    INBX Form 4 reported 4 separate P transactions for the same reporting
    owner, split across two custodial accounts and two dates) — a
    filing-level "was this a purchase?" boolean is not sufficient; the
    field dictionary's insider_buyer_count needs a per-REPORTING-OWNER
    count, and cluster_insider_buying_within_3mo needs to look across
    filings and owners, not just within one document.
  - nonDerivativeTable and derivativeTable are separate, independent
    tables. Open-market common stock purchases are non-derivative
    transactions; the field dictionary's insider-buying fields are about
    common stock purchases, not option/RSU grants, so only
    nonDerivativeTable is parsed here — derivativeTable is deliberately
    ignored, not a gap.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from lxml import etree

# Only these transactionCode values represent an actual open-market
# purchase. See module docstring for the full set of codes confirmed live.
PURCHASE_CODE = "P"


class Form4ParseError(Exception):
    pass


@dataclass
class Form4Transaction:
    document_id: str                    # accession number or URL, for provenance
    issuer_cik: str
    issuer_ticker: Optional[str]
    reporting_owner_cik: str
    reporting_owner_name: str
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    officer_title: Optional[str]
    security_title: str
    transaction_date: str               # ISO date
    transaction_code: str                # 'P' | 'S' | 'A' | 'M' | 'F' | 'G' | ...
    acquired_or_disposed: str            # 'A' | 'D' — independent of transaction_code, cross-check
    shares: Optional[float]
    price_per_share: Optional[float]
    shares_owned_after: Optional[float]
    ownership_nature: str                # 'D' (direct) | 'I' (indirect)
    indirect_ownership_note: Optional[str]  # e.g. "By Child A" — distinguishes sub-accounts

    @property
    def is_open_market_purchase(self) -> bool:
        """
        The one check that matters for insider_buying_within_3mo. Requires
        BOTH transaction_code == 'P' AND acquired_or_disposed == 'A' —
        these are two independently-reported fields in the XML; a filing
        where they disagree (confirmed structurally possible, not observed
        live) should not be silently trusted as a purchase.
        """
        return self.transaction_code == PURCHASE_CODE and self.acquired_or_disposed == "A"


def _text(el, path: str) -> Optional[str]:
    found = el.find(path)
    if found is None or found.text is None:
        return None
    return found.text.strip() or None


def _float(el, path: str) -> Optional[float]:
    raw = _text(el, path)
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _bool(el, path: str) -> bool:
    raw = _text(el, path)
    return raw == "1"


def parse_form4_xml(xml_bytes: bytes, document_id: str = "") -> List[Form4Transaction]:
    """
    Parse a raw Form 4 XML document into one Form4Transaction per reported
    non-derivative transaction (a filing may contain zero, one, or several
    — confirmed live up to 4 in a single document). derivativeTable is
    intentionally not parsed; see module docstring.

    document_id: caller-supplied identifier (e.g. accession number) carried
    through onto every transaction for provenance — this module has no
    knowledge of the surrounding documents table.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as e:
        raise Form4ParseError(f"Could not parse Form 4 XML: {e}") from e

    issuer = root.find("issuer")
    issuer_cik = _text(issuer, "issuerCik") or ""
    issuer_ticker = _text(issuer, "issuerTradingSymbol")

    owner = root.find("reportingOwner")
    owner_id = owner.find("reportingOwnerId") if owner is not None else None
    owner_cik = _text(owner_id, "rptOwnerCik") or ""
    owner_name = _text(owner_id, "rptOwnerName") or ""

    rel = owner.find("reportingOwnerRelationship") if owner is not None else None
    is_director = _bool(rel, "isDirector") if rel is not None else False
    is_officer = _bool(rel, "isOfficer") if rel is not None else False
    is_ten_pct = _bool(rel, "isTenPercentOwner") if rel is not None else False
    officer_title = _text(rel, "officerTitle") if rel is not None else None

    transactions: List[Form4Transaction] = []
    non_deriv = root.find("nonDerivativeTable")
    if non_deriv is None:
        return transactions

    for tx in non_deriv.findall("nonDerivativeTransaction"):
        coding = tx.find("transactionCoding")
        amounts = tx.find("transactionAmounts")
        post_amounts = tx.find("postTransactionAmounts")
        nature = tx.find("ownershipNature")

        transactions.append(Form4Transaction(
            document_id=document_id,
            issuer_cik=issuer_cik,
            issuer_ticker=issuer_ticker,
            reporting_owner_cik=owner_cik,
            reporting_owner_name=owner_name,
            is_director=is_director,
            is_officer=is_officer,
            is_ten_percent_owner=is_ten_pct,
            officer_title=officer_title,
            security_title=_text(tx, "securityTitle/value") or "",
            transaction_date=_text(tx, "transactionDate/value") or "",
            transaction_code=_text(coding, "transactionCode") or "",
            acquired_or_disposed=_text(amounts, "transactionAcquiredDisposedCode/value") or "",
            shares=_float(amounts, "transactionShares/value"),
            price_per_share=_float(amounts, "transactionPricePerShare/value"),
            shares_owned_after=_float(post_amounts, "sharesOwnedFollowingTransaction/value") if post_amounts is not None else None,
            ownership_nature=_text(nature, "directOrIndirectOwnership/value") or "",
            indirect_ownership_note=_text(nature, "natureOfOwnership/value") if nature is not None else None,
        ))

    return transactions


@dataclass
class InsiderBuyingSummary:
    """
    Aggregation result for insider_buying_within_3mo /
    cluster_insider_buying_within_3mo / insider_buyer_count — computed
    across ALL Form 4 filings discovered for a transaction's 3-month
    post-distribution window, not from a single filing.
    """
    any_buying: bool
    distinct_buyer_count: int             # distinct reporting_owner_cik with >=1 purchase
    cluster_threshold: int
    is_cluster: bool                       # distinct_buyer_count >= cluster_threshold
    purchase_transactions: List[Form4Transaction]


def summarize_insider_buying(
    all_transactions: List[Form4Transaction],
    window_start: str,
    window_end: str,
    cluster_threshold: int = 3,
) -> InsiderBuyingSummary:
    """
    Filter to genuine open-market purchases within [window_start,
    window_end] (inclusive, ISO dates) and summarize.

    cluster_threshold defaults to 3 distinct buyers — a configurable
    parameter, not a hardcoded assumption; the field dictionary flags this
    as a judgment call to confirm with Rich, not a fixed rule (see
    cluster_insider_buying_within_3mo in field_data_dictionary.py).

    distinct_buyer_count counts distinct reporting_owner_cik, NOT distinct
    transactions — confirmed live that one owner can have multiple
    purchase transactions in one filing (split across custodial sub-
    accounts) or across multiple filings; that must count as ONE buyer,
    not several, for both insider_buyer_count and the cluster threshold.
    """
    start_dt = datetime.strptime(window_start, "%Y-%m-%d")
    end_dt = datetime.strptime(window_end, "%Y-%m-%d")

    purchases = []
    for tx in all_transactions:
        if not tx.is_open_market_purchase:
            continue
        if not tx.transaction_date:
            continue
        tx_dt = datetime.strptime(tx.transaction_date, "%Y-%m-%d")
        if start_dt <= tx_dt <= end_dt:
            purchases.append(tx)

    distinct_buyers = {tx.reporting_owner_cik for tx in purchases}
    buyer_count = len(distinct_buyers)

    return InsiderBuyingSummary(
        any_buying=buyer_count > 0,
        distinct_buyer_count=buyer_count,
        cluster_threshold=cluster_threshold,
        is_cluster=buyer_count >= cluster_threshold,
        purchase_transactions=purchases,
    )
