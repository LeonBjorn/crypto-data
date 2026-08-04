"""The live broker's guards, which are the only part of it worth testing here.

Whether ccxt can talk to Hyperliquid is ccxt's problem and cannot be checked
without the network, which this suite forbids. What can be checked -- and is the
part that matters -- is that this module refuses to trade in every circumstance
it promises to refuse in.

Every test below drives a fake venue. That is not a compromise: the failures
being guarded against are this project's own mistakes, not the exchange's.
"""

import pytest

from paper.broker import BUY, SELL, BrokerError
from paper.hyperliquid import (
    ADDRESS_ENV,
    KEY_ENV,
    MAINNET_ENV,
    HyperliquidBroker,
    LiveTradingRefused,
)


class FakeVenue:
    """A venue that fills as instructed, so the guards can be tested alone."""

    def __init__(self, *, filled=None, average=100.0, fee=0.05, positions=(), status="closed",
                 markets=None, step=1e-5):
        self.markets = markets if markets is not None else {
            "BTC/USDC:USDC": {"swap": True, "base": "BTC", "precision": {"amount": step}},
            "BTC/USDC": {"swap": False, "base": "BTC", "precision": {"amount": step}},
            "ETH/USDC:USDC": {"swap": True, "base": "ETH", "precision": {"amount": step}},
            "ADA/USDC:USDC": {"swap": True, "base": "ADA", "precision": {"amount": 1.0}},
        }
        self.step = step
        self.filled = filled
        self.average = average
        self.fee = fee
        self._positions = list(positions)
        self.status = status
        self.sent = []

    def load_markets(self):
        return self.markets

    def amount_to_precision(self, symbol, amount):
        # Rounded through Decimal: int(0.01 / 1e-5) is 999, not 1000, and a fake
        # that loses precision would fail tests about code that does not.
        from decimal import Decimal
        step = self.markets.get(symbol, {}).get("precision", {}).get("amount", self.step)
        units = (Decimal(str(amount)) / Decimal(str(step))).to_integral_value(rounding="ROUND_FLOOR")
        return str(units * Decimal(str(step)))

    def create_order(self, symbol, type, side, amount, price=None, params=None, **kwargs):
        self.sent.append({"symbol": symbol, "side": side, "amount": amount,
                          "params": params or {}})
        return {
            "symbol": symbol, "side": side,
            "filled": amount if self.filled is None else self.filled,
            "average": self.average, "status": self.status,
            "fee": {"cost": self.fee}, "timestamp": 1_700_000_000_000,
        }

    def fetch_positions(self):
        return self._positions

    def fetch_balance(self):
        return {"free": {"USDC": 1000.0}, "total": {"USDC": 1000.0}}


def broker(**kwargs):
    kwargs.setdefault("client", FakeVenue())
    kwargs.setdefault("kill_switch", "does/not/exist")
    return HyperliquidBroker(**kwargs)


class TestItDoesNotTradeUnlessToldTwice:
    def test_dry_run_is_the_default(self):
        assert broker().dry_run is True

    def test_testnet_is_the_default(self):
        assert broker().network == "testnet"

    def test_a_dry_run_sends_nothing(self):
        venue = FakeVenue()
        b = broker(client=venue)
        b.market("BTC/USDT", BUY, 0.01, 100.0, 0)
        assert venue.sent == []

    def test_mainnet_needs_the_environment_variable_as_well(self, monkeypatch):
        monkeypatch.delenv(MAINNET_ENV, raising=False)
        with pytest.raises(LiveTradingRefused, match=MAINNET_ENV):
            broker(network="mainnet")

    def test_and_the_variable_alone_is_not_enough(self, monkeypatch):
        """The argument still has to say mainnet. Two switches, because one is
        one thing to get wrong.
        """
        monkeypatch.setenv(MAINNET_ENV, "yes")
        assert broker().network == "testnet"

    def test_with_both_it_is_allowed(self, monkeypatch):
        monkeypatch.setenv(MAINNET_ENV, "yes")
        assert broker(network="mainnet").network == "mainnet"

    def test_a_nonsense_network_is_refused(self):
        with pytest.raises(LiveTradingRefused):
            broker(network="production")


class TestTheKillSwitch:
    def test_its_presence_refuses_every_order(self, tmp_path):
        stop = tmp_path / "STOP"
        stop.write_text("", encoding="utf-8")
        b = broker(kill_switch=str(stop), dry_run=False)
        with pytest.raises(LiveTradingRefused, match="kill switch"):
            b.market("BTC/USDT", BUY, 0.01, 100.0, 0)

    def test_it_stops_a_dry_run_too(self, tmp_path):
        """A dry run that ignored the switch would report orders it would have
        placed while the operator believed everything was stopped.
        """
        stop = tmp_path / "STOP"
        stop.write_text("", encoding="utf-8")
        with pytest.raises(LiveTradingRefused):
            broker(kill_switch=str(stop)).market("BTC/USDT", BUY, 0.01, 100.0, 0)

    def test_removing_it_resumes(self, tmp_path):
        stop = tmp_path / "STOP"
        stop.write_text("", encoding="utf-8")
        b = broker(kill_switch=str(stop))
        stop.unlink()
        assert b.market("BTC/USDT", BUY, 0.01, 100.0, 0).qty == pytest.approx(0.01)


class TestTheCaps:
    def test_a_symbol_outside_the_allow_list_is_refused(self):
        b = broker(allowed_symbols={"BTC/USDT"})
        with pytest.raises(LiveTradingRefused, match="allowed list"):
            b.market("DOGE/USDT", BUY, 1.0, 1.0, 0)

    def test_an_empty_allow_list_permits_anything(self):
        """So the guard is opt-in rather than a silent block on a fresh setup."""
        assert broker().market("ETH/USDT", BUY, 0.01, 10.0, 0)

    def test_an_order_above_the_per_order_cap_is_refused(self):
        b = broker(max_order_notional=50.0)
        with pytest.raises(LiveTradingRefused, match="per-order cap"):
            b.market("BTC/USDT", BUY, 1.0, 100.0, 0)

    def test_it_refuses_rather_than_quietly_trading_less(self):
        """Truncating to the cap would mean the position taken is not the
        position the strategy asked for, and nothing downstream would know.
        """
        venue = FakeVenue()
        b = broker(client=venue, max_order_notional=50.0, dry_run=False)
        with pytest.raises(LiveTradingRefused):
            b.market("BTC/USDT", BUY, 1.0, 100.0, 0)
        assert venue.sent == []

    def test_an_order_breaching_total_exposure_is_refused(self):
        venue = FakeVenue(positions=[{"symbol": "ETH/USDC:USDC", "notional": 450.0,
                                      "contracts": 1.0, "side": "long"}])
        b = broker(client=venue, max_total_exposure=500.0, max_order_notional=1000.0)
        with pytest.raises(LiveTradingRefused, match="total exposure"):
            b.market("BTC/USDT", BUY, 1.0, 100.0, 0)

    def test_exposure_that_cannot_be_read_refuses_rather_than_assuming_zero(self):
        class Broken(FakeVenue):
            def fetch_positions(self):
                raise RuntimeError("venue unreachable")

        b = broker(client=Broken(), dry_run=False)
        with pytest.raises(LiveTradingRefused, match="cannot be checked"):
            b.market("BTC/USDT", BUY, 0.01, 100.0, 0)


class TestItReadsFillsBackRatherThanAssuming:
    def test_a_partial_fill_is_reported_as_partial(self):
        """The reason the seam was built this way. Assuming the requested size
        leaves the engine holding a position that does not exist.
        """
        venue = FakeVenue(filled=0.004)
        fill = broker(client=venue, dry_run=False).market("BTC/USDT", BUY, 0.01, 100.0, 0)
        assert fill.qty == pytest.approx(0.004)

    def test_the_price_is_the_venues_average_not_the_reference(self):
        venue = FakeVenue(average=101.5)
        fill = broker(client=venue, dry_run=False).market("BTC/USDT", BUY, 1.0, 100.0, 0)
        assert fill.price == pytest.approx(101.5)

    def test_the_fee_worsens_the_effective_price_in_the_direction_that_hurts(self):
        venue = FakeVenue(average=100.0, fee=1.0)
        buy = broker(client=venue, dry_run=False).market("BTC/USDT", BUY, 1.0, 100.0, 0)
        sell = broker(client=venue, dry_run=False).market("BTC/USDT", SELL, 1.0, 100.0, 0)
        assert buy.effective_price > buy.price
        assert sell.effective_price < sell.price

    def test_an_order_that_reports_no_fill_raises_rather_than_inventing_one(self):
        venue = FakeVenue(filled=0.0)
        with pytest.raises(BrokerError, match="no fill"):
            broker(client=venue, dry_run=False).market("BTC/USDT", BUY, 1.0, 100.0, 0)

    def test_positions_come_from_the_venue_not_from_memory(self):
        venue = FakeVenue(positions=[
            {"symbol": "BTC/USDC:USDC", "contracts": 0.5, "side": "long", "notional": 50.0},
            {"symbol": "ETH/USDC:USDC", "contracts": 2.0, "side": "short", "notional": 40.0},
        ])
        held = broker(client=venue).positions()
        assert held["BTC/USDC:USDC"] == pytest.approx(0.5)
        assert held["ETH/USDC:USDC"] == pytest.approx(-2.0)


class TestThingsItCannotDoAtAll:
    def test_there_is_no_withdrawal_method(self):
        """Absent rather than disabled. A method that does not exist cannot be
        called by a bug, a typo, or a future refactor.
        """
        for name in dir(HyperliquidBroker):
            assert "withdraw" not in name.lower()
            assert "transfer" not in name.lower()

    def test_credentials_are_not_accepted_as_arguments(self):
        """They come from the environment only, so a key cannot arrive in the
        repository by being pasted into a config file.
        """
        import inspect
        params = inspect.signature(HyperliquidBroker.__init__).parameters
        for forbidden in ("private_key", "privateKey", "secret", "wallet_address", "api_key"):
            assert forbidden not in params

    def test_live_without_credentials_refuses(self, monkeypatch):
        monkeypatch.delenv(ADDRESS_ENV, raising=False)
        monkeypatch.delenv(KEY_ENV, raising=False)
        with pytest.raises(LiveTradingRefused, match=ADDRESS_ENV):
            HyperliquidBroker(dry_run=False, client=None)

    def test_it_says_what_it_is_configured_as(self):
        assert "testnet" in broker().describe()
        assert "dry run" in broker().describe()


class TestTranslatingToTheVenue:
    """The store says BTC/USDT; Hyperliquid trades BTC/USDC:USDC."""

    def test_a_store_symbol_resolves_to_the_perpetual(self):
        assert broker().resolve("BTC/USDT") == "BTC/USDC:USDC"

    def test_it_does_not_fall_back_to_spot(self):
        """Both BTC/USDC and BTC/USDC:USDC exist on this venue and they are
        different instruments. Silently trading the wrong one is the failure the
        resolver exists to prevent.
        """
        assert broker().resolve("BTC/USDT") != "BTC/USDC"

    def test_a_base_with_no_perpetual_is_refused(self):
        """XRP has a perpetual on mainnet and none on testnet, so this is a real
        case rather than a hypothetical one.
        """
        with pytest.raises(LiveTradingRefused, match="no perpetual"):
            broker().resolve("XRP/USDT")

    def test_an_explicit_mapping_wins(self):
        b = broker(symbol_map={"BTC/USDT": "ETH/USDC:USDC"})
        assert b.resolve("BTC/USDT") == "ETH/USDC:USDC"

    def test_the_order_goes_to_the_venue_name(self):
        venue = FakeVenue()
        broker(client=venue, dry_run=False).market("BTC/USDT", BUY, 0.01, 100.0, 0)
        assert venue.sent[0]["symbol"] == "BTC/USDC:USDC"

    def test_but_the_fill_comes_back_under_the_name_that_was_asked_for(self):
        """The ledger is keyed the way the store and every earlier trade are."""
        fill = broker(client=FakeVenue(), dry_run=False).market("BTC/USDT", BUY, 0.01, 100.0, 0)
        assert fill.symbol == "BTC/USDT"


class TestQuantityRounding:
    def test_it_rounds_to_the_venues_lot_size(self):
        venue = FakeVenue(step=1e-5)
        broker(client=venue, dry_run=False).market("BTC/USDT", BUY, 0.0312345678, 100.0, 0)
        assert venue.sent[0]["amount"] == pytest.approx(0.03123, abs=1e-9)

    def test_a_quantity_that_rounds_to_zero_is_refused(self):
        """ADA's step is one whole unit, so a fifth of a unit becomes nothing.
        An order for nothing is not a small order; it reads afterwards as a
        filled position of size zero.
        """
        b = broker(client=FakeVenue(), dry_run=False, max_order_notional=1e9)
        with pytest.raises(LiveTradingRefused, match="rounds to zero"):
            b.market("ADA/USDT", BUY, 0.2, 1.0, 0)

    def test_nothing_is_sent_when_it_would_round_away(self):
        venue = FakeVenue()
        b = broker(client=venue, dry_run=False, max_order_notional=1e9)
        with pytest.raises(LiveTradingRefused):
            b.market("ADA/USDT", BUY, 0.2, 1.0, 0)
        assert venue.sent == []


class TestClosingIsReduceOnly:
    def test_close_sends_reduce_only(self):
        """A sell meant to close a long opens a short if the position has already
        gone. Reduce-only makes the worst case "nothing happened".
        """
        venue = FakeVenue()
        broker(client=venue, dry_run=False).close("BTC/USDT", SELL, 0.01, 100.0, 0)
        assert venue.sent[0]["params"].get("reduceOnly") is True

    def test_an_ordinary_order_is_not_reduce_only(self):
        venue = FakeVenue()
        broker(client=venue, dry_run=False).market("BTC/USDT", BUY, 0.01, 100.0, 0)
        assert not venue.sent[0]["params"].get("reduceOnly")


class TestCredentialsFromTheEnvironment:
    """Whitespace in an environment variable is the easiest mistake here.

    A trailing newline on the address produces a 422 from the venue reading
    "failed to deserialize the JSON body", which names neither the field nor the
    whitespace, and surfaces six frames down a traceback that ends in this
    module refusing to trade. It cost a real debugging session; it is stripped
    and validated now.
    """

    KEY = "0x" + "b" * 64
    ADDRESS = "0x" + "a" * 40

    def _env(self, monkeypatch, address):
        monkeypatch.setenv(ADDRESS_ENV, address)
        monkeypatch.setenv(KEY_ENV, self.KEY)

    @pytest.mark.parametrize("suffix", ["", "\n", " ", "\t", "\r\n"])
    def test_surrounding_whitespace_is_stripped(self, monkeypatch, suffix):
        self._env(monkeypatch, self.ADDRESS + suffix)
        broker = HyperliquidBroker(dry_run=False, network="testnet")
        assert broker._client.walletAddress == self.ADDRESS

    def test_the_key_is_stripped_too(self, monkeypatch):
        """A newline on the key breaks signing rather than deserialisation, so
        it fails later and even less legibly.
        """
        monkeypatch.setenv(ADDRESS_ENV, self.ADDRESS)
        monkeypatch.setenv(KEY_ENV, self.KEY + "\n")
        broker = HyperliquidBroker(dry_run=False, network="testnet")
        assert broker._client.privateKey == self.KEY

    @pytest.mark.parametrize("bad", [
        "a" * 40,                      # no 0x
        "0x" + "a" * 39,               # too short
        "0x" + "a" * 41,               # too long
        "0x" + "z" * 40,               # not hex
        "not-an-address",
    ])
    def test_a_malformed_address_is_refused_before_any_call(self, monkeypatch, bad):
        self._env(monkeypatch, bad)
        with pytest.raises(LiveTradingRefused, match="well-formed address"):
            HyperliquidBroker(dry_run=False, network="testnet")

    def test_the_refusal_says_what_to_look_for(self, monkeypatch):
        self._env(monkeypatch, "0xdeadbeef")
        with pytest.raises(LiveTradingRefused, match="space or newline"):
            HyperliquidBroker(dry_run=False, network="testnet")


class TestAnUnreadableExposureExplainsItself:
    def test_the_venues_own_error_is_carried_in_the_message(self):
        """Not only chained. The reason sat six frames down a traceback ending
        in this refusal, which reads as the guard being the fault.
        """
        class Broken(FakeVenue):
            def fetch_positions(self):
                raise RuntimeError("422 Unprocessable Entity")

        b = broker(client=Broken(), dry_run=False)
        with pytest.raises(LiveTradingRefused, match="422 Unprocessable Entity"):
            b.market("BTC/USDT", BUY, 0.01, 100.0, 0)
