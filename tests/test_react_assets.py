"""Python-only static UI delivery; no model load or Node server required."""
import unittest
from stockrl.web.resources import DASHBOARD_PATH, ASSET_NAMES, dashboard_asset

class ReactAssetChecks(unittest.TestCase):
    def test_built_entrypoint(self):
        page = DASHBOARD_PATH.read_text(encoding="utf-8")
        self.assertIn('id="root"', page)
        self.assertIn('/assets/index-', page)
        self.assertNotIn('web_dashboard.html', page)

    def test_assets_and_traversal(self):
        self.assertTrue(ASSET_NAMES)
        for name in ASSET_NAMES:
            body, mime = dashboard_asset('/assets/' + name)
            self.assertIsInstance(body, bytes)
            self.assertTrue(body)
            self.assertTrue(mime.startswith('text/'))
        self.assertIsNone(dashboard_asset('/assets/../resources.py'))
        self.assertIsNone(dashboard_asset('/assets/missing.js'))
        self.assertIsNone(dashboard_asset('/assets/C:/Windows/win.ini'))
