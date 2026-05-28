import unittest

from us_stock_money.model_config import THEME_BASKETS, WATCHLIST_TICKERS


class ModelConfigTests(unittest.TestCase):
    def test_selected_watchlist_tickers_are_in_theme_universe(self):
        requested = set(WATCHLIST_TICKERS)
        universe = {ticker for basket in THEME_BASKETS.values() for ticker in basket["tickers"]}
        self.assertLessEqual(requested, universe)
        self.assertIn("NOK", requested)


if __name__ == "__main__":
    unittest.main()
