import pytest
from unittest.mock import MagicMock, patch
from tools.url_context import URLContextTool

# Check if selenium is available
try:
    import selenium
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
class TestURLContextSelenium:
    
    @pytest.fixture
    def url_tool(self):
        return URLContextTool(use_llm_processing=False)

    def test_selenium_available(self):
        """Test that Selenium is correctly detected."""
        from tools.url_context import SELENIUM_AVAILABLE
        assert SELENIUM_AVAILABLE is True

    @patch('tools.url_context.webdriver.Chrome')
    @patch('tools.url_context.ChromeDriverManager')
    @patch('tools.url_context.Service')
    def test_extract_with_selenium_mock(self, mock_service, mock_manager, mock_chrome, url_tool):
        """Test extraction with mocked Selenium driver."""
        # Setup mock driver
        mock_driver = MagicMock()
        mock_chrome.return_value = mock_driver
        mock_driver.page_source = "<html><body><h1>Test Header</h1><p>Test content.</p></body></html>"
        
        # Mock find_element for fallback extraction
        mock_element = MagicMock()
        mock_element.text = "Test Header\nTest content."
        mock_driver.find_element.return_value = mock_element

        # Mock _process_content_with_llm
        with patch.object(url_tool, '_process_content_with_llm') as mock_process:
            mock_process.return_value = {"content": "Processed Content", "method": "llm_summary"}
            
            # Execute
            result = url_tool.extract_with_selenium("https://example.com")

            # Verify
            assert result["success"] is True
            assert result["content"] == "Processed Content"
            assert result["method"] == "selenium+llm_summary"
            mock_driver.get.assert_called_with("https://example.com")
            mock_driver.quit.assert_called_once()
            mock_process.assert_called_once()

    def test_extract_content_with_selenium_flag(self, url_tool):
        """Test that use_selenium flag triggers selenium extraction."""
        with patch.object(url_tool, 'extract_with_selenium') as mock_extract:
            mock_extract.return_value = {"success": True, "content": "Selenium Content", "method": "selenium"}
            
            # Call with use_selenium=True
            result = url_tool.extract_content(
                "https://example.com", 
                use_gemini=False, 
                use_selenium=True
            )
            
            assert result["success"] is True
            assert result["content"] == "Selenium Content"
            mock_extract.assert_called_once()
