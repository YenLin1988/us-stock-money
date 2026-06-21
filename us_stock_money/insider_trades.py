"""Recent corporate insider open-market trades from SEC Form 4 filings."""

from __future__ import annotations

import time
from urllib.parse import urljoin
from xml.etree import ElementTree

import pandas as pd
import requests


SEC_FORM4_FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=4&owner=only&count=100&output=atom"
)
SEC_USER_AGENT = "YenLin1988 us-stock-money admin@us-stock-money.local"
SUPPORTED_FORM_TYPES = {"4", "4/A"}

DISPLAY_COLUMNS = [
    "transaction_date",
    "filing_time",
    "ticker",
    "issuer_name",
    "owner_name",
    "role",
    "trade_side",
    "shares",
    "price_per_share",
    "estimated_value",
    "shares_after",
    "filing_url",
]


def download_insider_trades(max_filings: int = 30) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }
    )
    feed_response = session.get(SEC_FORM4_FEED_URL, timeout=30)
    feed_response.raise_for_status()

    filings = parse_form4_feed(feed_response.content, max_filings=max_filings)
    rows: list[dict[str, object]] = []
    for filing in filings:
        index_url = str(filing["filing_url"]).rsplit("/", 1)[0] + "/index.json"
        index_response = session.get(index_url, timeout=30)
        index_response.raise_for_status()
        xml_name = find_ownership_xml(index_response.json())
        if not xml_name:
            continue

        xml_url = urljoin(index_url, xml_name)
        xml_response = session.get(xml_url, timeout=30)
        xml_response.raise_for_status()
        rows.extend(
            parse_ownership_document(
                xml_response.content,
                filing_url=str(filing["filing_url"]),
                filing_time=str(filing["filing_time"]),
            )
        )
        time.sleep(0.12)

    return normalize_insider_trades(rows)


def parse_form4_feed(content: bytes, max_filings: int = 30) -> list[dict[str, str]]:
    root = ElementTree.fromstring(content)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    filings = []
    seen_urls = set()
    for entry in root.findall("atom:entry", namespace):
        link = entry.find("atom:link", namespace)
        updated = entry.findtext("atom:updated", default="", namespaces=namespace)
        filing_url = "" if link is None else str(link.attrib.get("href", ""))
        if not filing_url or filing_url in seen_urls:
            continue
        seen_urls.add(filing_url)
        filings.append({"filing_url": filing_url, "filing_time": updated})
        if len(filings) >= max_filings:
            break
    return filings


def find_ownership_xml(index_payload: dict[str, object]) -> str | None:
    directory = index_payload.get("directory", {})
    if not isinstance(directory, dict):
        return None
    items = directory.get("item", [])
    if not isinstance(items, list):
        return None
    xml_names = [
        str(item.get("name", ""))
        for item in items
        if isinstance(item, dict) and str(item.get("name", "")).lower().endswith(".xml")
    ]
    ownership_names = [name for name in xml_names if "ownership" in name.lower()]
    candidates = ownership_names or xml_names
    return candidates[0] if candidates else None


def parse_ownership_document(
    content: bytes,
    *,
    filing_url: str,
    filing_time: str,
) -> list[dict[str, object]]:
    root = ElementTree.fromstring(content)
    if _text(root, "documentType").upper() not in SUPPORTED_FORM_TYPES:
        return []

    issuer_name = _text(root, "issuer/issuerName")
    ticker = _text(root, "issuer/issuerTradingSymbol").upper()
    owner_name = _text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
    role = _owner_role(root)

    rows = []
    for transaction in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(transaction, "transactionCoding/transactionCode").upper()
        if code not in {"P", "S"}:
            continue
        shares = _number(transaction, "transactionAmounts/transactionShares/value")
        price = _number(transaction, "transactionAmounts/transactionPricePerShare/value")
        rows.append(
            {
                "transaction_date": _text(transaction, "transactionDate/value"),
                "filing_time": filing_time,
                "ticker": ticker,
                "issuer_name": issuer_name,
                "owner_name": owner_name,
                "role": role,
                "trade_side": "Purchase" if code == "P" else "Sale",
                "shares": shares,
                "price_per_share": price,
                "estimated_value": shares * price,
                "shares_after": _number(transaction, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
                "filing_url": filing_url,
            }
        )
    return rows


def normalize_insider_trades(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    for column in DISPLAY_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="coerce")
    frame["filing_time"] = pd.to_datetime(frame["filing_time"], errors="coerce", utc=True)
    for column in ["shares", "price_per_share", "estimated_value", "shares_after"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return (
        frame.dropna(subset=["transaction_date"])
        .drop_duplicates(
            subset=[
                "transaction_date",
                "ticker",
                "owner_name",
                "trade_side",
                "shares",
                "price_per_share",
                "shares_after",
            ]
        )
        .sort_values(["filing_time", "transaction_date"], ascending=False)
        .reset_index(drop=True)
    )


def filter_insider_trades(
    frame: pd.DataFrame,
    *,
    sides: list[str] | None = None,
    ticker: str = "",
) -> pd.DataFrame:
    filtered = frame.copy()
    if sides:
        filtered = filtered[filtered["trade_side"].isin(sides)]
    if ticker.strip():
        filtered = filtered[filtered["ticker"].str.contains(ticker.strip().upper(), regex=False)]
    return filtered.reset_index(drop=True)


def summarize_insider_trades(frame: pd.DataFrame) -> dict[str, float]:
    purchases = frame[frame["trade_side"] == "Purchase"]
    sales = frame[frame["trade_side"] == "Sale"]
    purchase_value = float(purchases["estimated_value"].sum())
    sale_value = float(sales["estimated_value"].sum())
    return {
        "trades": float(len(frame)),
        "tickers": float(frame.loc[frame["ticker"] != "", "ticker"].nunique()),
        "purchase_shares": float(purchases["shares"].sum()),
        "sale_shares": float(sales["shares"].sum()),
        "purchase_value": purchase_value,
        "sale_value": sale_value,
        "net_value": purchase_value - sale_value,
    }


def aggregate_insider_by_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "issuer_name",
        "signal",
        "trade_count",
        "purchases",
        "sales",
        "buy_value",
        "sale_value",
        "net_value",
        "insider_count",
        "latest_trade",
    ]
    usable = frame[frame["ticker"].fillna("") != ""].copy()
    if usable.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for ticker, group in usable.groupby("ticker", sort=False):
        purchases = group[group["trade_side"] == "Purchase"]
        sales = group[group["trade_side"] == "Sale"]
        buy_value = float(purchases["estimated_value"].sum())
        sale_value = float(sales["estimated_value"].sum())
        rows.append(
            {
                "ticker": ticker,
                "issuer_name": _latest_text(group, "issuer_name"),
                "signal": _activity_signal(buy_value, sale_value),
                "trade_count": len(group),
                "purchases": len(purchases),
                "sales": len(sales),
                "buy_value": buy_value,
                "sale_value": sale_value,
                "net_value": buy_value - sale_value,
                "insider_count": group["owner_name"].nunique(),
                "latest_trade": group["transaction_date"].max(),
            }
        )
    result = pd.DataFrame(rows)
    result["_activity"] = result["buy_value"] + result["sale_value"]
    return result.sort_values("_activity", ascending=False).drop(columns="_activity").reset_index(drop=True)


def _owner_role(root: ElementTree.Element) -> str:
    relationship = root.find("reportingOwner/reportingOwnerRelationship")
    if relationship is None:
        return ""
    roles = []
    if _is_true(relationship, "isDirector"):
        roles.append("Director")
    if _is_true(relationship, "isOfficer"):
        roles.append(_text(relationship, "officerTitle") or "Officer")
    if _is_true(relationship, "isTenPercentOwner"):
        roles.append("10% Owner")
    if _is_true(relationship, "isOther"):
        roles.append(_text(relationship, "otherText") or "Other")
    return ", ".join(roles)


def _is_true(root: ElementTree.Element, path: str) -> bool:
    return _text(root, path).lower() in {"1", "true", "yes"}


def _text(root: ElementTree.Element, path: str) -> str:
    value = root.findtext(path)
    return "" if value is None else value.strip()


def _number(root: ElementTree.Element, path: str) -> float:
    try:
        return float(_text(root, path).replace(",", ""))
    except ValueError:
        return 0.0


def _latest_text(frame: pd.DataFrame, column: str) -> str:
    values = frame.sort_values("transaction_date", ascending=False)[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else ""


def _activity_signal(buy_value: float, sale_value: float) -> str:
    total = buy_value + sale_value
    if total == 0:
        return "No Value"
    buy_share = buy_value / total
    if buy_share >= 0.65:
        return "Net Buying"
    if buy_share <= 0.35:
        return "Net Selling"
    return "Mixed"
