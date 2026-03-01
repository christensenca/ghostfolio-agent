#!/usr/bin/env python3
"""Generate mock Ghostfolio import JSON files for 3 investment strategies.

Fetches real historical prices from Yahoo Finance and CoinGecko,
then outputs deterministic fixture files for testing and eval.

Usage:
    pip install yfinance requests
    python generate_mock_data.py
"""
from __future__ import annotations

import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
PRICE_CACHE_FILE = OUTPUT_DIR / "price_cache.json"

# Deterministic seed
random.seed(42)

# Deterministic UUIDs via uuid5 (namespace + name)
_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")
ACCOUNT_BUY_HOLD_ID = str(uuid.uuid5(_NS, "buy-and-hold"))
ACCOUNT_WEEKLY_ETF_ID = str(uuid.uuid5(_NS, "weekly-etf"))
ACCOUNT_CRYPTO_ID = str(uuid.uuid5(_NS, "crypto-max"))

# Date range: March 3 2025 → Feb 23 2026
START_DATE = datetime(2025, 3, 3)
END_DATE = datetime(2026, 2, 23)


def load_price_cache() -> dict:
    if PRICE_CACHE_FILE.exists():
        with open(PRICE_CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_price_cache(cache: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRICE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_yahoo_prices(symbol: str, start: str, end: str, cache: dict) -> dict[str, float]:
    """Fetch daily close prices from Yahoo Finance. Returns {date_str: price}."""
    cache_key = f"yahoo:{symbol}:{start}:{end}"
    if cache_key in cache:
        print(f"  [cache] {symbol}")
        return cache[cache_key]

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    print(f"  [fetch] {symbol} from Yahoo Finance...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end)
    prices = {}
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        prices[date_str] = round(float(row["Close"]), 2)

    cache[cache_key] = prices
    return prices


def fetch_coingecko_prices(coin_id: str, start: str, end: str, cache: dict) -> dict[str, float]:
    """Fetch daily prices from CoinGecko. Returns {date_str: price}."""
    cache_key = f"coingecko:{coin_id}:{start}:{end}"
    if cache_key in cache:
        print(f"  [cache] {coin_id}")
        return cache[cache_key]

    import requests
    import time

    print(f"  [fetch] {coin_id} from CoinGecko...")
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prices = {}
    for ts_ms, price in data.get("prices", []):
        date_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        prices[date_str] = round(price, 2)

    cache[cache_key] = prices
    # CoinGecko rate limit: ~10-30 req/min for free tier
    time.sleep(2)
    return prices


def nearest_price(prices: dict[str, float], target_date: datetime) -> float:
    """Find the nearest price to a target date."""
    target_str = target_date.strftime("%Y-%m-%d")
    if target_str in prices:
        return prices[target_str]

    # Search nearby dates (up to 7 days in either direction)
    for delta in range(1, 8):
        for direction in [1, -1]:
            nearby = (target_date + timedelta(days=delta * direction)).strftime("%Y-%m-%d")
            if nearby in prices:
                return prices[nearby]

    # Fallback: return the closest date we have
    sorted_dates = sorted(prices.keys())
    if not sorted_dates:
        raise ValueError(f"No prices available near {target_str}")
    # Find closest
    closest = min(sorted_dates, key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d") - target_date).days))
    return prices[closest]


def make_activity(
    account_id: str,
    symbol: str,
    data_source: str,
    activity_type: str,
    date: datetime,
    quantity: float,
    unit_price: float,
    fee: float,
    currency: str = "USD",
    comment: str = "",
) -> dict:
    return {
        "accountId": account_id,
        "currency": currency,
        "dataSource": data_source,
        "date": date.strftime("%Y-%m-%dT00:00:00.000Z"),
        "fee": round(fee, 2),
        "quantity": round(quantity, 6),
        "symbol": symbol,
        "type": activity_type,
        "unitPrice": round(unit_price, 2),
        **({"comment": comment} if comment else {}),
    }


def generate_buy_and_hold(cache: dict) -> dict:
    """Strategy 1: Lump-sum buy of 4 stocks, hold all year."""
    print("\n--- Buy & Hold Portfolio ---")

    stocks = [
        {"symbol": "AAPL", "qty": 20},
        {"symbol": "MSFT", "qty": 15},
        {"symbol": "GOOGL", "qty": 25},
        {"symbol": "JNJ", "qty": 30},
    ]

    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    # Fetch prices
    all_prices = {}
    for stock in stocks:
        all_prices[stock["symbol"]] = fetch_yahoo_prices(
            stock["symbol"], start_str, end_str, cache
        )

    activities = []
    buy_date = datetime(2025, 3, 5)  # Wednesday

    # Initial buys
    for stock in stocks:
        price = nearest_price(all_prices[stock["symbol"]], buy_date)
        fee = round(random.uniform(0.50, 2.00), 2)
        activities.append(
            make_activity(
                ACCOUNT_BUY_HOLD_ID,
                stock["symbol"],
                "YAHOO",
                "BUY",
                buy_date,
                stock["qty"],
                price,
                fee,
                comment=f"Initial buy - Buy & Hold strategy",
            )
        )

    # Quarterly dividends for AAPL, MSFT, JNJ
    dividend_schedule = [
        # (symbol, date, per_share_amount)
        ("AAPL", datetime(2025, 5, 15), 0.26),
        ("AAPL", datetime(2025, 8, 14), 0.26),
        ("AAPL", datetime(2025, 11, 14), 0.26),
        ("MSFT", datetime(2025, 6, 12), 0.75),
        ("MSFT", datetime(2025, 9, 11), 0.75),
        ("MSFT", datetime(2025, 12, 11), 0.75),
        ("JNJ", datetime(2025, 6, 3), 1.24),
        ("JNJ", datetime(2025, 9, 2), 1.24),
        ("JNJ", datetime(2025, 12, 2), 1.24),
    ]

    qty_map = {s["symbol"]: s["qty"] for s in stocks}
    for symbol, div_date, per_share in dividend_schedule:
        if div_date > END_DATE:
            continue
        qty = qty_map[symbol]
        total_div = per_share * qty
        activities.append(
            make_activity(
                ACCOUNT_BUY_HOLD_ID,
                symbol,
                "YAHOO",
                "DIVIDEND",
                div_date,
                qty,
                per_share,
                0,
                comment=f"Quarterly dividend",
            )
        )

    return {
        "accounts": [
            {
                "id": ACCOUNT_BUY_HOLD_ID,
                "name": "Buy & Hold Portfolio",
                "currency": "USD",
                "balance": 0,
                "platformId": None,
            }
        ],
        "activities": sorted(activities, key=lambda a: a["date"]),
    }


def generate_weekly_etf(cache: dict) -> dict:
    """Strategy 2: Weekly $200 DCA into VOO."""
    print("\n--- Weekly ETF DCA ---")

    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    voo_prices = fetch_yahoo_prices("VOO", start_str, end_str, cache)

    activities = []
    current_date = START_DATE

    # Weekly buys (every Monday)
    while current_date <= END_DATE:
        # Find next Monday
        days_until_monday = (7 - current_date.weekday()) % 7
        if days_until_monday == 0 and current_date != START_DATE:
            buy_date = current_date
        else:
            buy_date = current_date + timedelta(days=days_until_monday)

        if buy_date > END_DATE:
            break

        price = nearest_price(voo_prices, buy_date)
        qty = round(200.0 / price, 6)  # $200 worth
        fee = 1.00

        activities.append(
            make_activity(
                ACCOUNT_WEEKLY_ETF_ID,
                "VOO",
                "YAHOO",
                "BUY",
                buy_date,
                qty,
                price,
                fee,
                comment="Weekly DCA",
            )
        )

        current_date = buy_date + timedelta(days=1)

    # Quarterly VOO dividends (roughly)
    total_shares_at = {}
    running_shares = 0.0
    for act in sorted(activities, key=lambda a: a["date"]):
        running_shares += act["quantity"]
        total_shares_at[act["date"][:10]] = running_shares

    voo_div_dates = [
        datetime(2025, 6, 27),
        datetime(2025, 9, 26),
        datetime(2025, 12, 22),
    ]
    voo_div_per_share = 1.60  # approximate quarterly dividend

    for div_date in voo_div_dates:
        if div_date > END_DATE:
            continue
        # Find shares held at dividend date
        div_date_str = div_date.strftime("%Y-%m-%d")
        shares_held = 0.0
        for date_str, shares in sorted(total_shares_at.items()):
            if date_str <= div_date_str:
                shares_held = shares
        if shares_held > 0:
            activities.append(
                make_activity(
                    ACCOUNT_WEEKLY_ETF_ID,
                    "VOO",
                    "YAHOO",
                    "DIVIDEND",
                    div_date,
                    round(shares_held, 6),
                    voo_div_per_share,
                    0,
                    comment="Quarterly dividend",
                )
            )

    return {
        "accounts": [
            {
                "id": ACCOUNT_WEEKLY_ETF_ID,
                "name": "Weekly ETF DCA",
                "currency": "USD",
                "balance": 0,
                "platformId": None,
            }
        ],
        "activities": sorted(activities, key=lambda a: a["date"]),
    }


def generate_crypto_max(cache: dict) -> dict:
    """Strategy 3: Aggressive crypto with buys and some profit-taking."""
    print("\n--- Crypto Portfolio ---")

    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")

    btc_prices = fetch_coingecko_prices("bitcoin", start_str, end_str, cache)
    eth_prices = fetch_coingecko_prices("ethereum", start_str, end_str, cache)

    activities = []

    # BTC buy schedule: larger buys early, smaller DCA later
    btc_buys = [
        (datetime(2025, 3, 5), 3000),   # $3000 initial
        (datetime(2025, 3, 20), 1000),
        (datetime(2025, 4, 15), 500),
        (datetime(2025, 5, 10), 500),
        (datetime(2025, 6, 15), 300),
        (datetime(2025, 7, 20), 300),
        (datetime(2025, 8, 15), 200),
        (datetime(2025, 9, 15), 200),
        (datetime(2025, 10, 15), 200),
        (datetime(2025, 11, 15), 200),
    ]

    for buy_date, usd_amount in btc_buys:
        if buy_date > END_DATE:
            continue
        price = nearest_price(btc_prices, buy_date)
        qty = round(usd_amount / price, 6)
        fee = round(random.uniform(0.50, 5.00), 2)
        activities.append(
            make_activity(
                ACCOUNT_CRYPTO_ID,
                "bitcoin",
                "COINGECKO",
                "BUY",
                buy_date,
                qty,
                price,
                fee,
                comment=f"BTC purchase - ${usd_amount}",
            )
        )

    # ETH buy schedule
    eth_buys = [
        (datetime(2025, 3, 7), 1500),   # $1500 initial
        (datetime(2025, 4, 10), 500),
        (datetime(2025, 5, 20), 300),
        (datetime(2025, 7, 1), 200),
        (datetime(2025, 9, 1), 200),
    ]

    for buy_date, usd_amount in eth_buys:
        if buy_date > END_DATE:
            continue
        price = nearest_price(eth_prices, buy_date)
        qty = round(usd_amount / price, 6)
        fee = round(random.uniform(0.50, 5.00), 2)
        activities.append(
            make_activity(
                ACCOUNT_CRYPTO_ID,
                "ethereum",
                "COINGECKO",
                "BUY",
                buy_date,
                qty,
                price,
                fee,
                comment=f"ETH purchase - ${usd_amount}",
            )
        )

    # Profit-taking sells
    # Sell ~10% of BTC in August (summer high?)
    btc_sell_date = datetime(2025, 8, 20)
    if btc_sell_date <= END_DATE:
        total_btc = sum(
            a["quantity"]
            for a in activities
            if a["symbol"] == "bitcoin" and a["type"] == "BUY" and a["date"] < btc_sell_date.strftime("%Y-%m-%dT00:00:00.000Z")
        )
        sell_qty = round(total_btc * 0.10, 6)
        sell_price = nearest_price(btc_prices, btc_sell_date)
        activities.append(
            make_activity(
                ACCOUNT_CRYPTO_ID,
                "bitcoin",
                "COINGECKO",
                "SELL",
                btc_sell_date,
                sell_qty,
                sell_price,
                round(random.uniform(1.00, 5.00), 2),
                comment="Taking 10% BTC profit",
            )
        )

    # Sell ~15% of ETH in December
    eth_sell_date = datetime(2025, 12, 15)
    if eth_sell_date <= END_DATE:
        total_eth = sum(
            a["quantity"]
            for a in activities
            if a["symbol"] == "ethereum" and a["type"] == "BUY" and a["date"] < eth_sell_date.strftime("%Y-%m-%dT00:00:00.000Z")
        )
        sell_qty = round(total_eth * 0.15, 6)
        sell_price = nearest_price(eth_prices, eth_sell_date)
        activities.append(
            make_activity(
                ACCOUNT_CRYPTO_ID,
                "ethereum",
                "COINGECKO",
                "SELL",
                eth_sell_date,
                sell_qty,
                sell_price,
                round(random.uniform(1.00, 5.00), 2),
                comment="Taking 15% ETH profit",
            )
        )

    return {
        "accounts": [
            {
                "id": ACCOUNT_CRYPTO_ID,
                "name": "Crypto Portfolio",
                "currency": "USD",
                "balance": 0,
                "platformId": None,
            }
        ],
        "activities": sorted(activities, key=lambda a: a["date"]),
    }


def summarize(data: dict, name: str) -> None:
    acts = data["activities"]
    buys = [a for a in acts if a["type"] == "BUY"]
    sells = [a for a in acts if a["type"] == "SELL"]
    divs = [a for a in acts if a["type"] == "DIVIDEND"]
    total_invested = sum(a["quantity"] * a["unitPrice"] for a in buys)
    total_fees = sum(a["fee"] for a in acts)
    symbols = set(a["symbol"] for a in acts)

    print(f"\n  {name}:")
    print(f"    Activities: {len(acts)} ({len(buys)} buys, {len(sells)} sells, {len(divs)} dividends)")
    print(f"    Symbols: {', '.join(sorted(symbols))}")
    print(f"    Total invested: ${total_invested:,.2f}")
    print(f"    Total fees: ${total_fees:,.2f}")
    print(f"    Date range: {acts[0]['date'][:10]} → {acts[-1]['date'][:10]}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_price_cache()

    print("Generating mock data with real historical prices...")

    buy_hold = generate_buy_and_hold(cache)
    weekly_etf = generate_weekly_etf(cache)
    crypto_max = generate_crypto_max(cache)

    # Save price cache
    save_price_cache(cache)

    # Write fixture files
    for name, data in [
        ("buy_and_hold", buy_hold),
        ("weekly_etf", weekly_etf),
        ("crypto_max", crypto_max),
    ]:
        filepath = OUTPUT_DIR / f"{name}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nWrote {filepath}")
        summarize(data, name)

    # Also generate a combined file with all 3 accounts
    combined = {
        "accounts": buy_hold["accounts"] + weekly_etf["accounts"] + crypto_max["accounts"],
        "activities": sorted(
            buy_hold["activities"] + weekly_etf["activities"] + crypto_max["activities"],
            key=lambda a: a["date"],
        ),
    }
    combined_path = OUTPUT_DIR / "all_accounts.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote {combined_path}")
    print(f"  Combined: {len(combined['activities'])} activities across {len(combined['accounts'])} accounts")

    print("\nDone! Run import_mock_data.py to load into Ghostfolio.")


if __name__ == "__main__":
    main()
