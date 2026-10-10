import unittest
import xml.etree.ElementTree as ET
from datetime import date
from unittest.mock import patch
from datetime import datetime, timezone

from src.Scripts.collect_laws import select_law, validate_body, xml_to_data, korea_today


class LawCollectionTests(unittest.TestCase):
    def test_default_date_tracks_today_in_korea(self):
        with patch('src.Scripts.collect_laws.datetime') as clock:
            clock.now.side_effect = lambda tz: datetime(2030, 1, 1, 16, tzinfo=timezone.utc).astimezone(tz)
            self.assertEqual(korea_today(), date(2030, 1, 2))

    def test_repeated_nodes_and_empty_fields_are_preserved(self):
        node = ET.fromstring('<법령 id="1"><조문><항>첫째</항><항>둘째</항></조문><빈값/></법령>')
        data = xml_to_data(node)
        self.assertEqual(data['조문']['항'], ['첫째', '둘째'])
        self.assertEqual(data['빈값'], '')
        self.assertEqual(data['@id'], '1')

    def test_exact_name_does_not_select_enforcement_decree(self):
        node = ET.fromstring('<LawSearch><law><법령명한글>근로기준법 시행령</법령명한글></law><law><법령명한글>근로기준법</법령명한글><법령ID>1</법령ID><법령일련번호>2</법령일련번호><시행일자>20250101</시행일자></law></LawSearch>')
        self.assertEqual(select_law(node, '근로기준법')['법령ID'], '1')
        with self.assertRaises(ValueError):
            select_law(node, '민법')

    def test_version_mismatch_and_future_laws_are_rejected(self):
        node = ET.fromstring('<법령><기본정보><법령ID>000001</법령ID><법령명_한글>민법</법령명_한글><시행일자>20270101</시행일자></기본정보><조문><조문단위><조문내용>제1조</조문내용></조문단위></조문></법령>')
        listing = {'법령ID': '1', '법령명한글': '민법', '시행일자': '20270101'}
        with self.assertRaises(ValueError):
            validate_body(node, listing, date(2026,10,10))
        with self.assertRaises(ValueError):
            validate_body(node, {**listing, '시행일자': '20260101'}, date(2028,1,1))
        self.assertEqual(validate_body(node, listing, date(2028,1,1))[1], 1)


if __name__ == '__main__':
    unittest.main()
