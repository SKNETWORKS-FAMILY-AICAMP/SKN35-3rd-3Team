import json
import unittest
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from src.Scripts.collect_precedents import sanitize_listing, valid_saved, collect


class PrecedentCollectionTests(unittest.TestCase):
    def test_explicit_ids_skip_search_and_preserve_invalid_existing_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            detail=ET.fromstring('<PrecService><판례정보일련번호>123</판례정보일련번호><판례내용>본문</판례내용></PrecService>')
            with patch('src.Scripts.collect_precedents.OUTPUT_DIR',folder), patch.dict('os.environ',{'LAW_API_OC':'test'}), \
                 patch('src.Scripts.collect_precedents.load_dotenv'), patch('src.Scripts.collect_precedents.requests.Session'), \
                 patch('src.Scripts.collect_precedents.time.sleep'), patch('src.Scripts.collect_precedents.request_xml',return_value=detail) as request:
                self.assertEqual(collect('unused',20,0,['123','123']),0)
                self.assertEqual(request.call_count,1)
                self.assertEqual(request.call_args.args[1],'lawService.do')
                raw=json.loads((folder/'123.json').read_text(encoding='utf-8'))
                self.assertEqual(raw['list_metadata'],{})
                self.assertIsNone(raw['query'])
                self.assertEqual(collect('unused',20,0,['123']),0)
                self.assertEqual(request.call_count,1)
                (folder/'456.json').write_text('broken raw',encoding='utf-8')
                self.assertEqual(collect('unused',20,0,['456']),1)
                self.assertEqual((folder/'456.json').read_text(encoding='utf-8'),'broken raw')
                self.assertEqual(request.call_count,1)
        with self.assertRaises(ValueError):
            collect('unused',20,0,['../123'])

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
