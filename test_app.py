import unittest
from unittest.mock import patch, mock_open, MagicMock, PropertyMock
import unicodedata # Import for creating test unicode strings if needed directly
import re # Import for creating test regex strings if needed directly
from app import normalize_text, process_file, extract_chapters_from_epub, PdfParser, handle_stop_button
from ebooklib import epub, ITEM_DOCUMENT # For type hints and constants like ITEM_DOCUMENT
# BeautifulSoup is imported within app.py, so we patch 'app.BeautifulSoup'
import fitz # For fitz.PyMuPDFError

class TestNormalizeText(unittest.TestCase):

    def test_simple_spaces_and_tabs(self):
        self.assertEqual(normalize_text("hello   world\t\there"), "hello world here")

    def test_leading_trailing_whitespace(self):
        self.assertEqual(normalize_text("  hello world  "), "hello world")

    def test_single_newlines_to_spaces(self):
        self.assertEqual(normalize_text("hello\nworld"), "hello world")

    def test_double_newlines_to_period_space(self):
        self.assertEqual(normalize_text("hello\n\nworld"), "hello . world")

    def test_triple_newlines_to_period_space(self):
        self.assertEqual(normalize_text("hello\n\n\nworld"), "hello . world")
        
    def test_quadruple_newlines_to_period_space(self):
        self.assertEqual(normalize_text("hello\n\n\n\nworld"), "hello . world")

    def test_mixed_newlines_and_spaces_complex(self):
        # Should result in "hello . world . another" after processing
        self.assertEqual(normalize_text("  hello \n\n world \n\n\n another \n part  "), "hello . world . another part")

    def test_windows_newlines_rn_single(self):
        self.assertEqual(normalize_text("hello\r\nworld"), "hello world")

    def test_windows_newlines_rn_multiple(self):
        self.assertEqual(normalize_text("hello\r\n\r\nworld"), "hello . world")

    def test_windows_newlines_rn_very_multiple(self):
        self.assertEqual(normalize_text("hello\r\n\r\n\r\nworld"), "hello . world")

    def test_nfkc_normalization_and_spacing(self):
        # Assuming NFKC converts full-width space and specific characters
        # For example, U+3000 is IDEOGRAPHIC SPACE, NFKC normalizes to U+0020 (SPACE)
        # And half-width katakana "ｶﾞ" (U+FF76 U+FF9E) to full-width "ガ" (U+30AC)
        test_str = "　テキスト　\n\nｶﾞギグゲゴ\n\n　" # Ideographic spaces, newlines, half-width kana
        expected_str = "テキスト . ガギグゲゴ ." # Normal spaces, period separator
        self.assertEqual(normalize_text(test_str), expected_str)

    def test_nfkc_with_multiple_spaces_and_newlines(self):
        # Half-width katakana "ﾊﾟ" (U+FF8A U+FF9F) to full-width "パ" (U+30D1)
        test_str = "  ﾊﾟ  \n\n  ﾊﾟ  "
        expected_str = "パ . パ"
        self.assertEqual(normalize_text(test_str), expected_str)

    def test_empty_string(self):
        self.assertEqual(normalize_text(""), "")

    def test_already_normalized_string(self):
        self.assertEqual(normalize_text("hello world ."), "hello world .")

    def test_string_with_only_spaces(self):
        self.assertEqual(normalize_text("   "), "")
        
    def test_string_with_only_newlines(self):
        self.assertEqual(normalize_text("\n\n"), ".") # \n\n -> " . ", then strip() -> "."

    def test_string_with_only_newlines_multiple(self):
        self.assertEqual(normalize_text("\n\n\n\n"), ".")

    def test_string_with_single_newline(self):
        self.assertEqual(normalize_text("\n"), "") # \n -> " ", then strip() -> ""

    def test_string_with_spaces_and_multiple_newlines(self):
        # "  \n\n  " -> "  .  " -> "."
        self.assertEqual(normalize_text("  \n\n  "), ".")

    def test_string_with_spaces_and_single_newline(self):
        # "  \n  " -> "    " -> ""
        self.assertEqual(normalize_text("  \n  "), "")
        
    def test_complex_mixed_spacing_and_newlines(self):
        self.assertEqual(normalize_text("  first part \n second part \n\n third part \n\n\n fourth part  "), "first part second part . third part . fourth part")

    def test_only_rn_sequences(self):
        self.assertEqual(normalize_text("\r\n\r\n"), ".")

    def test_rn_followed_by_n(self):
        # \r\n -> \n, so effectively \n\n\n -> .
        self.assertEqual(normalize_text("\r\n\n\n"), ".")
        
    def test_text_with_no_alphanum_just_newlines_and_spaces(self):
        self.assertEqual(normalize_text("  \n  \n\n  \n  "), ".")

class TestProcessFileTXT(unittest.TestCase):
    def setUp(self):
        # Create a mock file object for Gradio file uploads
        self.mock_file_obj = MagicMock()
        self.mock_file_obj.name = "test.txt" # Ensure it's treated as a TXT file

    @patch('builtins.open', new_callable=mock_open)
    def test_simple_utf8_txt(self, mocked_open):
        mock_content = "Hello UTF-8 World!"
        mocked_open.return_value.read.return_value = mock_content
        
        result = process_file(self.mock_file_obj)
        
        mocked_open.assert_called_once_with(self.mock_file_obj.name, 'r', encoding='utf-8')
        self.assertEqual(result, [{'title': 'Full Document', 'text': 'Hello UTF-8 World!'}])

    @patch('builtins.open')
    def test_utf8_fails_latin1_succeeds(self, mocked_open):
        mock_content_latin1 = "Hällö Wörld" # Correctly represents latin-1 characters
        
        # Side effect function for mock_open
        def open_side_effect(path, mode='r', encoding=None):
            if encoding == 'utf-8':
                # Simulate read failing for utf-8 by raising UnicodeDecodeError
                m_file = mock_open()() # Get the file handle mock
                m_file.read.side_effect = UnicodeDecodeError('utf-8', b'\x00', 0, 1, 'reason')
                return m_file
            elif encoding == 'latin-1':
                return mock_open(read_data=mock_content_latin1)()
            else:
                # Fallback for any other call, though not expected in this test path
                return mock_open(read_data="default content")()

        mocked_open.side_effect = open_side_effect
        
        result = process_file(self.mock_file_obj)
        
        self.assertEqual(mocked_open.call_count, 2)
        mocked_open.assert_any_call(self.mock_file_obj.name, 'r', encoding='utf-8')
        mocked_open.assert_any_call(self.mock_file_obj.name, 'r', encoding='latin-1')
        self.assertEqual(result, [{'title': 'Full Document', 'text': 'Hällö Wörld'}])


    @patch('builtins.open')
    def test_utf8_and_latin1_fail_on_decode(self, mocked_open):
        def open_side_effect(path, mode='r', encoding=None):
            m_file = mock_open()()
            m_file.read.side_effect = UnicodeDecodeError(encoding, b'\x00', 0, 1, 'reason')
            return m_file

        mocked_open.side_effect = open_side_effect
        
        result = process_file(self.mock_file_obj)
        
        self.assertEqual(mocked_open.call_count, 2) # utf-8 and latin-1 attempts
        self.assertEqual(result, [{'title': 'Error: TXT file encoding issue (tried UTF-8, latin-1)', 'text': ''}])

    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_file_not_found(self, mocked_open):
        result = process_file(self.mock_file_obj)
        mocked_open.assert_called_once_with(self.mock_file_obj.name, 'r', encoding='utf-8')
        self.assertEqual(result, [{'title': 'Error: TXT file not found', 'text': ''}])

    @patch('builtins.open')
    def test_io_error_on_read(self, mocked_open):
        # Configure the first (utf-8) attempt to raise IOError on read
        m_file_utf8 = mock_open()()
        m_file_utf8.read.side_effect = IOError("Disk read error")
        mocked_open.return_value = m_file_utf8 # For the first call
        
        result = process_file(self.mock_file_obj)
        
        mocked_open.assert_called_once_with(self.mock_file_obj.name, 'r', encoding='utf-8')
        self.assertEqual(result, [{'title': 'Error: Could not read TXT file (IO error)', 'text': ''}])

    @patch('builtins.open', new_callable=mock_open, read_data="")
    def test_empty_txt_file(self, mocked_open):
        result = process_file(self.mock_file_obj)
        mocked_open.assert_called_once_with(self.mock_file_obj.name, 'r', encoding='utf-8')
        self.assertEqual(result, [{'title': 'Full Document', 'text': ''}])

    @patch('builtins.open', side_effect=RuntimeError("Some generic error on open"))
    def test_general_exception_on_open(self, mocked_open):
        # This tests the outermost catch-all 'Exception' in the TXT processing block
        result = process_file(self.mock_file_obj)
        mocked_open.assert_called_once_with(self.mock_file_obj.name, 'r', encoding='utf-8')
        self.assertEqual(result, [{'title': 'Error: Unexpected error processing TXT file', 'text': ''}])

    @patch('builtins.open')
    def test_latin1_read_fails_with_non_unicode_exception(self, mocked_open):
        # UTF-8 attempt fails with UnicodeDecodeError
        # Latin-1 attempt fails with a different Exception (e.g., IOError)
        def open_side_effect(path, mode='r', encoding=None):
            if encoding == 'utf-8':
                m_file_utf8 = mock_open()()
                m_file_utf8.read.side_effect = UnicodeDecodeError('utf-8', b'\x00', 0, 1, 'reason')
                return m_file_utf8
            elif encoding == 'latin-1':
                m_file_latin1 = mock_open()()
                m_file_latin1.read.side_effect = IOError("Failed to read latin-1")
                return m_file_latin1
            return mock_open()() 

        mocked_open.side_effect = open_side_effect
        result = process_file(self.mock_file_obj)
        self.assertEqual(mocked_open.call_count, 2)
        self.assertEqual(result, [{'title': 'Error: Could not read TXT file with fallback encoding', 'text': ''}])


class TestExtractChaptersEPUB(unittest.TestCase):
    def _setup_mock_item(self, name="item.xhtml", content="<p>Default content</p>", item_type=ITEM_DOCUMENT):
        mock_item = MagicMock(spec=epub.EpubHtml) # Use EpubHtml as a common spec for document items
        mock_item.get_type.return_value = item_type
        mock_item.get_body_content.return_value = content.encode('utf-8') # Encode to bytes
        mock_item.file_name = name
        return mock_item

    def _setup_mock_soup(self, mock_bs_constructor, text_content="Default text", find_result=None):
        mock_soup_instance = MagicMock()
        mock_soup_instance.get_text.return_value = text_content
        if find_result: # if find_result is a mock element
            mock_soup_instance.find.return_value = find_result
        else: # if find_result is None (element not found)
            mock_soup_instance.find.return_value = None
        mock_bs_constructor.return_value = mock_soup_instance
        return mock_soup_instance

    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_successful_toc_extraction_simple_links(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book

        mock_toc_link1 = MagicMock(spec=epub.Link, title="Chapter 1", href="chap1.xhtml")
        mock_toc_link2 = MagicMock(spec=epub.Link, title="Chapter 2", href="chap2.xhtml")
        mock_book.toc = [mock_toc_link1, mock_toc_link2]

        item1_content = "<p>Content of Chapter 1</p>"
        item2_content = "<p>Content of Chapter 2</p>"
        mock_item1 = self._setup_mock_item(name="chap1.xhtml", content=item1_content)
        mock_item2 = self._setup_mock_item(name="chap2.xhtml", content=item2_content)
        
        def get_item_side_effect(href):
            if href == "chap1.xhtml": return mock_item1
            if href == "chap2.xhtml": return mock_item2
            return None
        mock_book.get_item_with_href.side_effect = get_item_side_effect
        
        # Mock BeautifulSoup for each call
        def bs_side_effect(html_content, parser):
            if html_content == item1_content.encode('utf-8'):
                return self._setup_mock_soup(MagicMock(), text_content="Content of Chapter 1") # new mock for BS
            elif html_content == item2_content.encode('utf-8'):
                return self._setup_mock_soup(MagicMock(), text_content="Content of Chapter 2")
            return MagicMock() # default
        mock_bs_constructor.side_effect = bs_side_effect

        chapters = extract_chapters_from_epub("dummy.epub")
        self.assertEqual(chapters, [
            {'title': 'Chapter 1', 'text': 'Content of Chapter 1'},
            {'title': 'Chapter 2', 'text': 'Content of Chapter 2'}
        ])
        self.assertEqual(mock_bs_constructor.call_count, 2)


    @patch('app.epub.read_epub')
    def test_corrupted_epub_file(self, mock_read_epub):
        mock_read_epub.side_effect = epub.EpubException("Failed to read EPUB")
        chapters = extract_chapters_from_epub("corrupted.epub")
        self.assertEqual(chapters, [])

    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_empty_toc_fallback_to_full_document(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book
        mock_book.toc = [] # Empty TOC

        mock_item1 = self._setup_mock_item(content="<p>Full text part 1.</p>")
        mock_item2 = self._setup_mock_item(content="<p>Full text part 2.</p>")
        mock_book.get_items_of_type.return_value = [mock_item1, mock_item2]
        
        # Setup BeautifulSoup to return specific text for specific content
        mock_soup1 = MagicMock()
        mock_soup1.get_text.return_value = "Full text part 1."
        mock_soup2 = MagicMock()
        mock_soup2.get_text.return_value = "Full text part 2."

        def bs_side_effect(html_content, parser):
            if html_content == b"<p>Full text part 1.</p>": return mock_soup1
            if html_content == b"<p>Full text part 2.</p>": return mock_soup2
            return MagicMock()
        mock_bs_constructor.side_effect = bs_side_effect

        chapters = extract_chapters_from_epub("dummy.epub")
        self.assertEqual(chapters, [{'title': 'Full Document', 'text': 'Full text part 1. Full text part 2.'}])
        mock_book.get_items_of_type.assert_called_once_with(ITEM_DOCUMENT)


    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_toc_entry_invalid_href(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book

        mock_toc_link_valid = MagicMock(spec=epub.Link, title="Chapter 1", href="chap1.xhtml")
        mock_toc_link_invalid = MagicMock(spec=epub.Link, title="Chapter 2", href="chap2_nonexistent.xhtml")
        mock_book.toc = [mock_toc_link_valid, mock_toc_link_invalid]

        mock_item1 = self._setup_mock_item(name="chap1.xhtml", content="<p>Valid Chapter</p>")
        mock_book.get_item_with_href.side_effect = lambda href: mock_item1 if href == "chap1.xhtml" else None
        
        self._setup_mock_soup(mock_bs_constructor, text_content="Valid Chapter")

        chapters = extract_chapters_from_epub("dummy.epub")
        self.assertEqual(chapters, [{'title': 'Chapter 1', 'text': 'Valid Chapter'}])


    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_chapter_content_with_fragment(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book

        mock_toc_link = MagicMock(spec=epub.Link, title="Section 1", href="chap1.xhtml#frag1")
        mock_book.toc = [mock_toc_link]

        mock_item = self._setup_mock_item(content="<p id='other'>Other</p><p id='frag1'>Fragment Text</p>")
        mock_book.get_item_with_href.return_value = mock_item
        
        mock_fragment_element = MagicMock()
        mock_fragment_element.get_text.return_value = "Fragment Text"
        self._setup_mock_soup(mock_bs_constructor, text_content="Full text if fragment fails", find_result=mock_fragment_element)

        chapters = extract_chapters_from_epub("dummy.epub")
        self.assertEqual(chapters, [{'title': 'Section 1', 'text': 'Fragment Text'}])
        mock_bs_constructor.return_value.find.assert_called_once_with(id='frag1')

    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_chapter_content_fragment_not_found(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book
        mock_toc_link = MagicMock(spec=epub.Link, title="Section 1", href="chap1.xhtml#frag_not_found")
        mock_book.toc = [mock_toc_link]
        mock_item = self._setup_mock_item(content="<p>Full Text Fallback</p>")
        mock_book.get_item_with_href.return_value = mock_item
        
        self._setup_mock_soup(mock_bs_constructor, text_content="Full Text Fallback", find_result=None)

        chapters = extract_chapters_from_epub("dummy.epub")
        self.assertEqual(chapters, [{'title': 'Section 1', 'text': 'Full Text Fallback'}])
        mock_bs_constructor.return_value.find.assert_called_once_with(id='frag_not_found')

    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_empty_epub_no_text_after_processing(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book
        mock_book.toc = []
        mock_item_empty = self._setup_mock_item(content="<p></p>") # Empty content
        mock_book.get_items_of_type.return_value = [mock_item_empty]
        self._setup_mock_soup(mock_bs_constructor, text_content="") # Soup returns empty text

        chapters = extract_chapters_from_epub("empty.epub")
        self.assertEqual(chapters, [{'title': 'Empty Document', 'text': ''}])

    @patch('app.epub.read_epub')
    def test_no_document_items_in_fallback(self, mock_read_epub):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book
        mock_book.toc = []
        mock_book.get_items_of_type.return_value = [] # No document items
        
        chapters = extract_chapters_from_epub("no_docs.epub")
        self.assertEqual(chapters, [{'title': 'Empty Document', 'text': ''}])

    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_nested_toc_structure_tuple_section_links(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book

        mock_section = MagicMock(spec=epub.Section, title="Part 1")
        mock_link1 = MagicMock(spec=epub.Link, title="Chapter 1.1", href="p1c1.xhtml")
        mock_link2 = MagicMock(spec=epub.Link, title="Chapter 1.2", href="p1c2.xhtml")
        mock_book.toc = [(mock_section, [mock_link1, mock_link2])]

        item_p1c1 = self._setup_mock_item(name="p1c1.xhtml", content="<p>P1C1 Content</p>")
        item_p1c2 = self._setup_mock_item(name="p1c2.xhtml", content="<p>P1C2 Content</p>")
        
        mock_book.get_item_with_href.side_effect = lambda href: item_p1c1 if href == "p1c1.xhtml" else item_p1c2 if href == "p1c2.xhtml" else None

        def bs_side_effect(html_content, parser):
            if html_content == b"<p>P1C1 Content</p>": return self._setup_mock_soup(MagicMock(), "P1C1 Content")
            if html_content == b"<p>P1C2 Content</p>": return self._setup_mock_soup(MagicMock(), "P1C2 Content")
            return MagicMock()
        mock_bs_constructor.side_effect = bs_side_effect
        
        chapters = extract_chapters_from_epub("nested.epub")
        expected = [
            {'title': 'Part 1 - Chapter 1.1', 'text': 'P1C1 Content'},
            {'title': 'Part 1 - Chapter 1.2', 'text': 'P1C2 Content'}
        ]
        self.assertEqual(chapters, expected)

    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_nested_toc_structure_section_with_children(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book

        mock_section = MagicMock(spec=epub.Section, title="Part 2")
        mock_link_child = MagicMock(spec=epub.Link, title="Chapter 2.1", href="p2c1.xhtml")
        # PropertyMock for 'children' as it might be accessed directly
        type(mock_section).children = PropertyMock(return_value=[mock_link_child])
        mock_book.toc = [mock_section]
        
        item_p2c1 = self._setup_mock_item(name="p2c1.xhtml", content="<p>P2C1 Content</p>")
        mock_book.get_item_with_href.return_value = item_p2c1
        self._setup_mock_soup(mock_bs_constructor, text_content="P2C1 Content")

        chapters = extract_chapters_from_epub("nested_children.epub")
        self.assertEqual(chapters, [{'title': 'Part 2 - Chapter 2.1', 'text': 'P2C1 Content'}])

    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_toc_link_with_no_title(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book
        mock_toc_link = MagicMock(spec=epub.Link, title=None, href="chap_no_title.xhtml")
        mock_book.toc = [mock_toc_link]
        mock_item = self._setup_mock_item(content="<p>Content</p>")
        mock_book.get_item_with_href.return_value = mock_item
        self._setup_mock_soup(mock_bs_constructor, text_content="Content")

        chapters = extract_chapters_from_epub("no_title.epub")
        self.assertEqual(chapters, [{'title': 'Chapter 1', 'text': 'Content'}]) # Default title "Chapter X"

    @patch('app.BeautifulSoup')
    @patch('app.epub.read_epub')
    def test_toc_item_not_link_or_section_or_tuple(self, mock_read_epub, mock_bs_constructor):
        mock_book = MagicMock()
        mock_read_epub.return_value = mock_book
        
        mock_toc_link_valid = MagicMock(spec=epub.Link, title="Chapter 1", href="chap1.xhtml")
        mock_book.toc = ["unexpected string item", mock_toc_link_valid] # Add an invalid item

        mock_item1 = self._setup_mock_item(name="chap1.xhtml", content="<p>Valid Chapter</p>")
        mock_book.get_item_with_href.return_value = mock_item1
        self._setup_mock_soup(mock_bs_constructor, text_content="Valid Chapter")

        # We expect the invalid item to be skipped and the valid one processed.
        # The function prints a warning for skipped items.
        with patch('builtins.print') as mock_print: # To check warnings
            chapters = extract_chapters_from_epub("mixed_toc.epub")
            self.assertEqual(chapters, [{'title': 'Chapter 1', 'text': 'Valid Chapter'}])
            # Check if a warning about the unexpected item was printed
            self.assertTrue(any("Skipping unknown or unhandled TOC entry type: <class 'str'>" in call.args[0] for call in mock_print.call_args_list))

class TestPdfParserGetChapters(unittest.TestCase):
    def setUp(self):
        self.mock_file_obj = MagicMock()
        self.mock_file_obj.name = "dummy.pdf"

    @patch('app.pymupdf4llm.to_markdown')
    @patch('app.fitz.open')
    def test_successful_toc_extraction(self, mock_fitz_open, mock_to_markdown):
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc
        
        mock_doc.get_toc.return_value = [[1, "Chapter 1", 1], [1, "Chapter 2", 3]]
        mock_doc.page_count = 4
        
        page_contents = {
            0: "Text C1P1", # Page 1 (0-indexed)
            1: "Text C1P2", # Page 2
            2: "Text C2P1"  # Page 3
        }
        mock_pages = {
            i: MagicMock(get_text=MagicMock(return_value=text)) 
            for i, text in page_contents.items()
        }
        mock_doc.load_page.side_effect = lambda idx: mock_pages.get(idx, MagicMock(get_text=MagicMock(return_value="")))
        
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()

        expected_chapters = [
            {'title': 'Chapter 1', 'text': 'Text C1P1Text C1P2'},
            {'title': 'Chapter 2', 'text': 'Text C2P1'}
        ]
        self.assertEqual(chapters, expected_chapters)
        mock_fitz_open.assert_called_once_with("dummy.pdf")
        mock_doc.get_toc.assert_called_once()
        mock_doc.load_page.assert_any_call(0) # page 1 for ch1
        mock_doc.load_page.assert_any_call(1) # page 2 for ch1
        mock_doc.load_page.assert_any_call(2) # page 3 for ch2
        mock_pages[0].get_text.assert_called_once_with("text")
        mock_pages[1].get_text.assert_called_once_with("text")
        mock_pages[2].get_text.assert_called_once_with("text")
        mock_doc.close.assert_called_once()
        mock_to_markdown.assert_not_called()

    @patch('app.fitz.open')
    def test_corrupted_pdf_open_error(self, mock_fitz_open):
        mock_fitz_open.side_effect = fitz.PyMuPDFError("Failed to open PDF")
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()
        self.assertEqual(chapters, [{'title': 'Error: Could not open or read PDF (PyMuPDF specific error)', 'text': ''}])
        mock_fitz_open.assert_called_once_with("dummy.pdf")

    @patch('app.fitz.open')
    def test_general_exception_on_open(self, mock_fitz_open):
        mock_fitz_open.side_effect = RuntimeError("Some other error")
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()
        self.assertEqual(chapters, [{'title': 'Error: Could not open or read PDF', 'text': ''}])
        mock_fitz_open.assert_called_once_with("dummy.pdf")

    @patch('app.pymupdf4llm.to_markdown')
    @patch('app.fitz.open')
    def test_no_toc_fallback_to_markdown_success(self, mock_fitz_open, mock_to_markdown):
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc
        mock_doc.get_toc.return_value = [] # No TOC
        mock_to_markdown.return_value = "Full document markdown text."
        
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()
        
        self.assertEqual(chapters, [{'title': 'Full Document', 'text': 'Full document markdown text.'}])
        mock_doc.get_toc.assert_called_once()
        mock_to_markdown.assert_called_once_with(mock_doc)
        mock_doc.close.assert_called_once()

    @patch('app.pymupdf4llm.to_markdown')
    @patch('app.fitz.open')
    def test_no_toc_markdown_extraction_fails(self, mock_fitz_open, mock_to_markdown):
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc
        mock_doc.get_toc.return_value = []
        mock_to_markdown.side_effect = Exception("Markdown conversion failed")
        
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()
        
        self.assertEqual(chapters, [{'title': 'Full Document (Markdown Extraction Failed)', 'text': ''}])
        mock_to_markdown.assert_called_once_with(mock_doc)
        mock_doc.close.assert_called_once()

    @patch('app.pymupdf4llm.to_markdown')
    @patch('app.fitz.open')
    def test_empty_pdf_no_text_from_toc_pages_empty_markdown(self, mock_fitz_open, mock_to_markdown):
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc
        mock_doc.get_toc.return_value = [[1, "Chapter 1", 1]]
        mock_doc.page_count = 1
        
        mock_page = MagicMock()
        mock_page.get_text.return_value = "" # Empty text from page
        mock_doc.load_page.return_value = mock_page
        
        mock_to_markdown.return_value = "" # Empty text from markdown fallback
        
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()
        
        self.assertEqual(chapters, [{'title': 'Empty Document', 'text': ''}])
        mock_doc.close.assert_called_once()

    @patch('app.pymupdf4llm.to_markdown')
    @patch('app.fitz.open')
    def test_empty_pdf_no_toc_empty_markdown(self, mock_fitz_open, mock_to_markdown):
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc
        mock_doc.get_toc.return_value = []
        mock_to_markdown.return_value = ""
        
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()
        
        self.assertEqual(chapters, [{'title': 'Empty Document', 'text': ''}])
        mock_doc.close.assert_called_once()

    @patch('app.pymupdf4llm.to_markdown')
    @patch('app.fitz.open')
    @patch('builtins.print') # To suppress and check warnings
    def test_toc_page_extraction_error(self, mock_print, mock_fitz_open, mock_to_markdown):
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc
        mock_doc.get_toc.return_value = [[1, "Chapter 1", 1]] # Chapter from page 1 to end
        mock_doc.page_count = 2

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Text from page 1."
        mock_page2 = MagicMock()
        mock_page2.get_text.side_effect = Exception("Failed to extract text from page 2")

        mock_doc.load_page.side_effect = lambda idx: mock_page1 if idx == 0 else mock_page2 if idx == 1 else MagicMock()
        
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()
        
        self.assertEqual(chapters, [{'title': 'Chapter 1', 'text': 'Text from page 1.'}])
        mock_doc.close.assert_called_once()
        # Check if the warning was printed for the failed page
        self.assertTrue(any("Warning: Could not extract text from page 2" in call.args[0] for call in mock_print.call_args_list if call.args))


    @patch('app.pymupdf4llm.to_markdown')
    @patch('app.fitz.open')
    def test_get_toc_fails_fallback_to_markdown(self, mock_fitz_open, mock_to_markdown):
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc
        mock_doc.get_toc.side_effect = Exception("Failed to get TOC")
        mock_to_markdown.return_value = "Fallback markdown content."
        
        parser = PdfParser(self.mock_file_obj)
        chapters = parser.get_chapters()
        
        self.assertEqual(chapters, [{'title': 'Full Document', 'text': 'Fallback markdown content.'}])
        mock_doc.get_toc.assert_called_once()
        mock_to_markdown.assert_called_once_with(mock_doc)
        mock_doc.close.assert_called_once()

class TestStopButtonHandler(unittest.TestCase):
    @patch('app.print') # Patch print where handle_stop_button can find it
    def test_handle_stop_button_logic(self, mock_print):
        streaming_active_val, out_stream_val = handle_stop_button()
        self.assertFalse(streaming_active_val)
        self.assertIsNone(out_stream_val)
        # Verify the debug print call
        mock_print.assert_called_once_with("DEBUG: Stop button Python handler: Setting streaming_active to False and out_stream to None.")

if __name__ == '__main__':
    unittest.main()
