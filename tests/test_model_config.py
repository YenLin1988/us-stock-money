import unittest

from us_stock_money.model_config import THEME_BASKETS


class ModelConfigTests(unittest.TestCase):
    def test_selected_watchlist_tickers_are_in_theme_universe(self):
        requested = {
            "BE", "GEV", "VRT", "RMBS", "PENG", "ONTO", "NOW", "NOK", "MRVL",
            "KULR", "KOPN", "IREN", "IRDM", "INTC", "FIG", "AVGO", "ASX", "AMD",
            "AMAT", "ALAB", "ALMU", "APP", "CRWD", "CRWV", "EOSE", "ETN", "GLW",
            "PL", "RDW", "RKLB", "SNDK", "SIDU", "VST", "AXTI", "NBIS", "OSS",
            "MP", "UUUU", "TSLA", "TSM", "AVAV", "RCAT", "RTX",
        }
        universe = {ticker for basket in THEME_BASKETS.values() for ticker in basket["tickers"]}
        self.assertLessEqual(requested, universe)


if __name__ == "__main__":
    unittest.main()
