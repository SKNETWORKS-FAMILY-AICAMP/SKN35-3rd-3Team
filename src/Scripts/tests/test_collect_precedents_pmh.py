import json
import unittest
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

from src.Scripts.collect_precedents import sanitize_listing, valid_saved


class PrecedentCollectionTests(unittest.TestCase):
    def test_authentication_removed_without_losing_link_parameters(self):
        listing = {"판례상세링크": "/DRF/lawService.do?OC=sample&ID=123&type=XML&blank=", "사건명": "사건"}
        safe = sanitize_listing(listing)
        params = parse_qs(urlsplit(safe["판례상세링크"]).query, keep_blank_values=True)
        self.assertEqual(params, {"ID": ["123"], "type": ["XML"], "blank": [""]})
        self.assertEqual(safe["사건명"], listing["사건명"])
        self.assertIn("OC=sample", listing["판례상세링크"])
        self.assertNotIn("oc=", sanitize_listing({"link": "https://example.com/?oc=sample"})["link"])

    def test_cache_rejects_wrong_internal_id_and_invalid_shapes(self):
        path = Mock()
        path.exists.return_value = True
        valid = {"precedent_id": "123", "detail": {"판례정보일련번호": "123", "판례내용": "본문"}}
        path.read_text.return_value = json.dumps(valid)
        self.assertTrue(valid_saved(path, "123"))
        invalid = [[], {}, {**valid, "detail": []},
                   {**valid, "detail": {**valid["detail"], "판례정보일련번호": "999"}},
                   {**valid, "detail": {**valid["detail"], "판례내용": 5}},
                   {**valid, "detail": {**valid["detail"], "판례내용": " "}}]
        for data in invalid:
            path.read_text.return_value = json.dumps(data)
            self.assertFalse(valid_saved(path, "123"))


if __name__ == "__main__":
    unittest.main()
