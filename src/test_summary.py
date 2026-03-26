import unittest
import web_summary
import json
import os
import tempfile

class TestSummaryIntegration(unittest.TestCase):

    def test_limit_words(self):
        text = "one two three four five"
        self.assertEqual(web_summary.limit_words(text, 3), "one two three")
        self.assertEqual(web_summary.limit_words(text, 10), "one two three four five")
        self.assertEqual(web_summary.limit_words("", 5), "")

    def test_clean_html_to_text_blocks(self):
        html = """
        <html>
            <body>
                <nav>Menu</nav>
                <h1>Title</h1>
                <p>Paragraph 1.</p>
                <script>console.log('test');</script>
                <footer>Footer</footer>
                <p>Paragraph 2.</p>
            </body>
        </html>
        """
        blocks = web_summary.clean_html_to_text_blocks(html)
        self.assertEqual(blocks, ["Title", "Paragraph 1.", "Paragraph 2."])

    def test_calculate_words_per_block(self):
        blocks = ["a", "b", "c"]
        self.assertEqual(web_summary.calculate_words_per_block(blocks, 30), 10)
        self.assertEqual(web_summary.calculate_words_per_block([], 30), 0)

    def test_prune_content(self):
        blocks = ["word " * 20, "word " * 20]
        pruned = web_summary.prune_content(blocks, 10)
        self.assertTrue(len(pruned.split()) <= 10)

    def test_parse_summary_json(self):
        result, err = web_summary._parse_summary_json('{"summary": "hello"}', "Test")
        self.assertEqual(result, {"summary": "hello"})
        self.assertIsNone(err)

        result, err = web_summary._parse_summary_json("no json here", "Test")
        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertIn("error", err)  # type: ignore[arg-type]

        result, err = web_summary._parse_summary_json("", "Test")
        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertIn("error", err)  # type: ignore[arg-type]

    def test_cache_roundtrip(self):
        # Use a temp DB so we don't pollute the real cache
        old_db = web_summary.CACHE_DB
        fd, tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        web_summary.CACHE_DB = tmp
        try:
            url, prompt, wc = "https://test.invalid", "test", 50
            self.assertIsNone(web_summary.cache_get(url, prompt, wc))

            web_summary.cache_put(url, prompt, wc, {"summary": "cached"})
            self.assertEqual(web_summary.cache_get(url, prompt, wc), {"summary": "cached"})

            # Different params should miss
            self.assertIsNone(web_summary.cache_get(url, "other", wc))
        finally:
            web_summary.CACHE_DB = old_db
            os.unlink(tmp)

    def test_real_pipeline(self):
        """Tests the full pipeline against example.com via arbiter or Ollama."""
        # Use temp cache DB
        old_db = web_summary.CACHE_DB
        fd, tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        web_summary.CACHE_DB = tmp
        try:
            url = "https://example.com"
            result_json_str = web_summary.run_pipeline(url, "Summarize this page.", word_count=50)

            data = json.loads(result_json_str)
            self.assertIn("summary", data, f"JSON result missing 'summary' key: {data}")

            summary_text = data["summary"]
            self.assertTrue(len(summary_text) > 0, "Summary text was empty")
            print(f"\n[LLM Response] Summary: {summary_text}")

            word_count = len(summary_text.split())
            print(f"Word count: {word_count}")
            self.assertTrue(word_count < 100, f"Summary too long: {word_count} words")

            # Second call should hit cache
            result2 = web_summary.run_pipeline(url, "Summarize this page.", word_count=50)
            self.assertEqual(result_json_str, result2)
        finally:
            web_summary.CACHE_DB = old_db
            os.unlink(tmp)

if __name__ == '__main__':
    unittest.main()