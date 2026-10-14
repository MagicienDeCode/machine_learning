#!/usr/bin/env python3
"""Fetch EUR exchange rates to RMB (CNY), JPY and USD.

Sources:
    ecb   Frankfurter API, European Central Bank daily reference rates (default)
    sina  Sina Finance spot FX quotes, near real-time market rates (unofficial endpoint)

Usage:
    python exchange_rate.py                      # rates for 1 EUR from the ECB
    python exchange_rate.py 250                  # convert 250 EUR
    python exchange_rate.py --source sina        # live market rates from Sina
    python exchange_rate.py --date 2024-01-15    # historical ECB rates
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

BASE = "EUR"
# RMB's ISO 4217 code is CNY
TARGETS = {"CNY": "RMB", "JPY": "JPY", "USD": "USD"}

FRANKFURTER_URL = "https://api.frankfurter.dev/v1"
OPEN_ER_URL = "https://open.er-api.com/v6/latest/"
SINA_URL = "https://hq.sinajs.cn/list="
# Sina rejects requests that don't come from its own site
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}


def _get(url, headers, timeout):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_ecb(date="latest", targets=TARGETS, timeout=10):
    """Return (as_of, {currency: rate}) for 1 EUR from the ECB via Frankfurter."""
    url = f"{FRANKFURTER_URL}/{date}?from={BASE}&to={','.join(targets)}"
    data = json.loads(_get(url, {"User-Agent": "exchange-rate-script"}, timeout))
    return data["date"], data["rates"]


def fetch_open_er(targets=TARGETS, timeout=10):
    """Return (as_of, {currency: rate}) for 1 EUR from open.er-api.com (daily, includes RUB)."""
    data = json.loads(_get(OPEN_ER_URL + BASE, {"User-Agent": "exchange-rate-script"}, timeout))
    if data.get("result") != "success":
        raise ValueError(f"open.er-api error: {data.get('error-type', 'unknown')}")
    return data["time_last_update_utc"], {code: data["rates"][code] for code in targets}


def fetch_sina(targets=TARGETS, timeout=10):
    """Return (as_of, {currency: rate}) for 1 EUR from Sina Finance spot quotes.

    Each quote looks like: var hq_str_fx_seurcny="22:29:57,bid,ask,prev_close,...,latest,name,...,date";
    Field 0 is the time, field 8 the latest price and the last field the date.
    """
    symbols = [f"fx_s{BASE.lower()}{code.lower()}" for code in targets]
    text = _get(SINA_URL + ",".join(symbols), SINA_HEADERS, timeout).decode("gbk")
    quotes = dict(re.findall(r'hq_str_(\w+)="([^"]*)"', text))

    rates, as_of = {}, None
    for code, symbol in zip(targets, symbols):
        fields = quotes.get(symbol, "").split(",")
        if len(fields) < 9 or not fields[8]:
            raise ValueError(f"no Sina quote for {BASE}/{code}")
        rates[code] = float(fields[8])
        as_of = f"{fields[-1]} {fields[0]} Beijing time"
    return as_of, rates


def main():
    parser = argparse.ArgumentParser(description="EUR exchange rates to RMB, JPY, USD")
    parser.add_argument("amount", nargs="?", type=float, default=1.0, help="amount in EUR (default 1)")
    parser.add_argument("--source", choices=["ecb", "sina"], default="ecb", help="rate source (default ecb)")
    parser.add_argument("--date", default="latest", help="historical date YYYY-MM-DD, ecb only (default latest)")
    args = parser.parse_args()

    if args.source == "sina" and args.date != "latest":
        parser.error("--date is only supported with --source ecb")

    try:
        if args.source == "sina":
            as_of, rates = fetch_sina()
        else:
            as_of, rates = fetch_ecb(args.date)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        print(f"Failed to fetch exchange rates from {args.source}: {e}", file=sys.stderr)
        return 1

    print(f"Exchange rates for {args.amount:,.2f} {BASE} ({args.source.upper()}, {as_of}):")
    for code, label in TARGETS.items():
        rate = rates[code]
        print(f"  {label:<4} {args.amount * rate:>14,.4f}   (1 {BASE} = {rate} {code})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
