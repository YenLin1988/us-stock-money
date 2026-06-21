"""Recent corporate insider open-market trades from SEC Form 4 filings."""

from __future__ import annotations

import time
import re
from urllib.parse import urljoin
from xml.etree import ElementTree

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


SEC_FORM4_FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=4&owner=only&count=100&output=atom"
)
SEC_USER_AGENT = "YenLin1988 us-stock-money admin@us-stock-money.local"
SUPPORTED_FORM_TYPES = {"4", "4/A"}
NOKIA_MANAGER_TRANSACTIONS_URL = "https://www.nokia.com/newsroom/?h=1&t=managers+transactions"

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
    "transaction_nature",
    "source",
    "currency",
    "filing_url",
]


def download_insider_trades(max_filings: int = 30) -> pd.DataFrame:
    frames = []
    try:
        frames.append(download_sec_form4_trades(max_filings=max_filings))
    except Exception:
        pass
    try:
        frames.append(download_nokia_manager_trades(max_releases=max_filings))
    except Exception:
        pass
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        raise RuntimeError("No SEC Form 4 or supplemental manager transaction data was available")
    return normalize_insider_trades(pd.concat(usable, ignore_index=True).to_dict("records"))


def download_sec_form4_trades(max_filings: int = 30) -> pd.DataFrame:
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


def download_nokia_manager_trades(max_releases: int = 30) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "us-stock-money/1.0"})
    response = session.get(NOKIA_MANAGER_TRANSACTIONS_URL, timeout=30)
    response.raise_for_status()
    links = parse_nokia_release_links(response.text, limit=max_releases)
    rows = []
    for url in links:
        release = session.get(url, timeout=30)
        release.raise_for_status()
        rows.extend(parse_nokia_manager_release(release.text, filing_url=url))
    return normalize_insider_trades(rows)


def download_ticker_insider_trades(ticker: str) -> pd.DataFrame:
    symbol = ticker.strip().upper()
    if not symbol:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    frames = []
    yahoo_rows = normalize_yahoo_insider_transactions(
        yf.Ticker(symbol).get_insider_transactions(),
        ticker=symbol,
    )
    if not yahoo_rows.empty:
        frames.append(yahoo_rows)
    if symbol == "NOK":
        try:
            frames.append(download_nokia_manager_trades())
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    return normalize_insider_trades(pd.concat(frames, ignore_index=True).to_dict("records"))


def normalize_yahoo_insider_transactions(
    frame: pd.DataFrame | None,
    *,
    ticker: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    rows = []
    for record in frame.to_dict("records"):
        text = str(record.get("Text") or "")
        side = _classify_yahoo_side(text)
        if side is None:
            continue
        shares = _coerce_number(record.get("Shares"))
        estimated_value = _coerce_number(record.get("Value"))
        price = estimated_value / shares if shares and estimated_value else _price_from_text(text)
        transaction_date = pd.to_datetime(record.get("Start Date"), errors="coerce")
        if pd.isna(transaction_date):
            continue
        rows.append(
            {
                "transaction_date": transaction_date,
                "filing_time": transaction_date,
                "ticker": ticker.upper(),
                "issuer_name": ticker.upper(),
                "owner_name": str(record.get("Insider") or ""),
                "role": str(record.get("Position") or ""),
                "trade_side": side,
                "shares": shares,
                "price_per_share": price,
                "estimated_value": estimated_value or shares * price,
                "shares_after": 0.0,
                "transaction_nature": "Purchase" if side == "Purchase" else "Sale",
                "source": "Yahoo Finance",
                "currency": "USD",
                "filing_url": str(record.get("URL") or ""),
            }
        )
    return normalize_insider_trades(rows)


def parse_nokia_release_links(content: str, limit: int = 30) -> list[str]:
    soup = BeautifulSoup(content, "html.parser")
    links = []
    seen = set()
    for anchor in soup.select("a.td_headlines[href]"):
        title = anchor.get("title", "")
        url = anchor.get("href", "")
        if "managers' transactions" not in str(title).lower() or not url or url in seen:
            continue
        seen.add(url)
        links.append(str(url))
        if len(links) >= limit:
            break
    return links


def parse_nokia_manager_release(
    content: str,
    *,
    filing_url: str,
) -> list[dict[str, object]]:
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text("\n", strip=True)
    start = text.find("Transaction notification under Article 19")
    end = text.find("About Nokia", start)
    if start < 0:
        return []
    section = text[start:end if end > start else None]
    owner_name = _match_text(section, r"Name:\s*([^\n]+)")
    role = _match_text(section, r"Position:\s*([^\n]+)")
    filing_time = _match_text(text, r"Managers[’'] transactions\s*(\d{1,2}\s+\w+\s+\d{4})")
    blocks = re.split(r"(?=Transaction date:\s*\d{4}-\d{2}-\d{2})", section)
    rows = []
    for block in blocks:
        transaction_date = _match_text(block, r"Transaction date:\s*(\d{4}-\d{2}-\d{2})")
        nature = _match_text(block, r"Nature of the transaction:\s*([^\n]+)").upper()
        if not transaction_date or nature not in {"ACQUISITION", "DISPOSAL"}:
            continue
        detail = re.search(
            r"Transaction details\s*\(1\):\s*Volume:\s*([\d\s,.]+)\s+Unit price:\s*([\d\s,.]+|N/A)(?:\s+([A-Z]{3}))?",
            block,
            flags=re.IGNORECASE,
        )
        if detail is None:
            continue
        shares = _parse_number(detail.group(1))
        price = _parse_number(detail.group(2))
        venue = _match_text(block, r"Venue:\s*([^\n]+)")
        currency = (detail.group(3) or ("USD" if "XNYS" in venue else "EUR")).upper()
        rows.append(
            {
                "transaction_date": transaction_date,
                "filing_time": filing_time,
                "ticker": "NOK",
                "issuer_name": "Nokia Corporation",
                "owner_name": owner_name,
                "role": role,
                "trade_side": "Purchase" if nature == "ACQUISITION" else "Sale",
                "shares": shares,
                "price_per_share": price,
                "estimated_value": shares * price,
                "shares_after": 0.0,
                "transaction_nature": nature.title(),
                "source": "Nokia Article 19",
                "currency": currency,
                "filing_url": filing_url,
            }
        )
    return rows


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
                "transaction_nature": "Open-market purchase" if code == "P" else "Open-market sale",
                "source": "SEC Form 4",
                "currency": "USD",
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
        "currencies",
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
                "currencies": ", ".join(sorted(group["currency"].dropna().astype(str).unique())),
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


def _match_text(content: str, pattern: str) -> str:
    match = re.search(pattern, content, flags=re.IGNORECASE)
    return "" if match is None else match.group(1).strip()


def _parse_number(value: str) -> float:
    cleaned = value.replace("\xa0", "").replace(" ", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _coerce_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if pd.notna(number) else 0.0


def _classify_yahoo_side(text: str) -> str | None:
    value = text.lower()
    if value.startswith("purchase") or " purchase at " in f" {value} ":
        return "Purchase"
    if value.startswith("sale") or " sale at " in f" {value} ":
        return "Sale"
    return None


def _price_from_text(text: str) -> float:
    match = re.search(r"price\s+([\d,.]+)(?:\s*-\s*([\d,.]+))?", text, flags=re.IGNORECASE)
    if match is None:
        return 0.0
    low = _parse_number(match.group(1))
    high = _parse_number(match.group(2)) if match.group(2) else low
    return (low + high) / 2


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
