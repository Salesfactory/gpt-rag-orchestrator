"""Unit tests for the HTML parser."""

import unittest

from webscrapping.config import CrawlerConfig
from webscrapping.html_parser import HtmlParser


class TestHtmlParser(unittest.TestCase):
    def _make_parser(self, striptags: bool = True) -> HtmlParser:
        return HtmlParser(
            CrawlerConfig(
                documents={"urls": []},
                html={
                    "striptags": striptags,
                    "parser": {"ignored_classes": []},
                },
            )
        )

    def test_custom_markdown_chunking_returns_text_chunks(self):
        parser = self._make_parser(striptags=True)
        chunks = parser.custom_markdown_chunking("# Title\n\nParagraph text")
        self.assertTrue(chunks)
        self.assertTrue(any("Title" in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
