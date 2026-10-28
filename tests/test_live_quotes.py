"""Cotización en vivo sin red: respuestas simuladas de Yahoo y gold-api."""
import requests
from metals import live_quotes

YAHOO = {"chart": {"result": [{"meta": {"regularMarketPrice": 4054.6, "chartPreviousClose": 4074.6, "regularMarketDayHigh": 4084.1,
                                        "regularMarketDayLow": 4040.6, "regularMarketTime": 1793207386},
                               "indicators": {"quote": [{"close": [4070.0, None, 4054.6]}]}}]}}
GOLD_API = {"price": 4040.6, "updatedAt": "2026-10-28T17:19:28Z"}


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        if isinstance(self.data, Exception):
            raise self.data

    def json(self):
        return self.data


class Session:
    def __init__(self, yahoo, gold_api):
        self.answers = {"yahoo": yahoo, "gold-api": gold_api}

    def get(self, url, **kwargs):
        return Response(self.answers["yahoo" if "yahoo" in url else "gold-api"])


def test_yahoo_gives_daily_change_and_intraday():
    q = live_quotes.quote("gold", Session(YAHOO, GOLD_API))
    assert q["source"].startswith("Yahoo") and q["price"] == 4054.6
    assert round(q["change"], 4) == round(4054.6 / 4074.6 - 1, 4)
    assert q["intraday"] == [4070.0, 4054.6]


def test_falls_back_to_gold_api_then_to_none():
    q = live_quotes.quote("silver", Session(requests.HTTPError("429"), GOLD_API))
    assert q["source"].startswith("gold-api") and q["change"] is None and q["time"].year == 2026
    assert live_quotes.quote("gold", Session({"chart": {"result": []}}, {"price": -1, "updatedAt": "2026-10-28T00:00:00Z"})) is None
