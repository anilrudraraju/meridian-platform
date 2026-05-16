"""
Stubs for SEC filing retrieval in the Special Situations Lab.

These delegate to the helpers already defined in app.py at import time.
Full implementations live in app.py to avoid circular imports; this module
provides a clean interface for the spinoff_lab UI to call.
"""
from typing import Tuple, Dict


def fetch_spinoff_10k(ticker: str, year: str = None) -> Tuple[bool, str, str, str, str]:
    """
    Retrieve the 10-K text for a spinoff company via SEC EDGAR.

    Returns (ok, text, description, cik, company_name).

    TODO: wire up to fetch_edgar_filing(ticker, "10-K", target_year=year)
          which is already implemented in app.py.
    """
    raise NotImplementedError(
        "Call fetch_edgar_filing(ticker, '10-K', target_year=year) from app.py directly."
    )


def fetch_spinoff_form10(ticker: str) -> Tuple[bool, str, str, str, str]:
    """
    Retrieve the Form 10 (registration statement) for a newly spun-off company.

    Returns (ok, text, description, cik, company_name).

    TODO: wire up to fetch_edgar_filing(ticker, "10-12B") in app.py.
          Form 10s use filing type "10-12B" on EDGAR.
    """
    raise NotImplementedError(
        "Call fetch_edgar_filing(ticker, '10-12B') from app.py directly."
    )


def fetch_spinoff_xbrl(ticker: str) -> Tuple[bool, Dict, str]:
    """
    Retrieve XBRL structured financial facts for a spinoff company.

    Returns (ok, facts_dict, message).

    TODO: wire up to fetch_xbrl_facts(ticker) from app.py.
    """
    raise NotImplementedError(
        "Call fetch_xbrl_facts(ticker) from app.py directly."
    )


def index_spinoff_filing(text: str, ticker: str, company_name: str,
                         cik: str, form_type: str, fiscal_year: str,
                         rag_system) -> int:
    """
    Chunk and index a spinoff filing into the provided RAGSystem instance.

    Returns the number of chunks indexed.

    TODO: instantiate DocumentProcessor, call process_filing(), then
          rag_system.index_documents(chunks). Both classes are in app.py.
    """
    raise NotImplementedError(
        "Instantiate DocumentProcessor and call process_filing() + "
        "rag_system.index_documents() from app.py directly."
    )
