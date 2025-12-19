import unittest
import web_summary
import json
import requests

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
        blocks = ["word " * 20, "word " * 20] # 20 words each
        # target 10 words total -> 5 per block
        pruned = web_summary.prune_content(blocks, 10)
        self.assertTrue(len(pruned.split()) <= 10)
        
    def test_real_ollama_pipeline(self):
        """
        Tests the full pipeline against the real 'example.com' and the local Ollama instance.
        """
        url = "https://example.com"
        # We expect a JSON string back, testing with 50 words summary
        result_json_str = web_summary.run_pipeline(url, "Summarize this page.", word_count=50)
        
        # Verify it parses as JSON
        try:
            data = json.loads(result_json_str)
        except json.JSONDecodeError:
            self.fail(f"Pipeline returned invalid JSON: {result_json_str}")
            
        # Verify structure
        self.assertIn("summary", data, "JSON result missing 'summary' key")
        # Properties should NOT be there (or we just ignore it, but user said remove it)
        # We didn't explicitly forbid it in schema, just didn't ask for it.
        # But let's check content exists
        
        summary_text = data["summary"]
        self.assertTrue(len(summary_text) > 0, "Summary text was empty")
        print(f"\n[Real Ollama Response] Summary: {summary_text}")
        
        # Basic check if word count is somewhat respected (it's approximate for LLMs)
        word_count = len(summary_text.split())
        print(f"Word count: {word_count}")
        # We asked for 50, so it shouldn't be huge.
        self.assertTrue(word_count < 100, f"Summary too long: {word_count} words")

if __name__ == '__main__':
    unittest.main()