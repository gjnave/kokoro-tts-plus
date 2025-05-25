from kokoro import KModel, KPipeline
import gradio as gr
import os
import random
import torch
import ebooklib
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import pymupdf as fitz
import pymupdf4llm
import json
import time
import unicodedata
import re

# Environment settings
IS_DUPLICATE = not os.getenv('SPACE_ID', '').startswith('hexgrad/')
CHAR_LIMIT = None if IS_DUPLICATE else 5000

# Model initialization
CUDA_AVAILABLE = torch.cuda.is_available()
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def normalize_text(text):
    """Normalize text by Unicode normalization, handling paragraph breaks, and stripping excess whitespace."""
    if not isinstance(text, str):
        # print(f"Warning: normalize_text received non-string input: {type(text)}. Coercing to string.")
        text = str(text)

    # 1. Apply Unicode normalization (NFKC)
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception as e:
        print(f"Warning: Unicode normalization failed for a text segment. Error: {e}")
        # Proceed with the original text if normalization fails

    # 2. Standardize Windows-style newlines to Unix-style
    text = text.replace('\r\n', '\n')

    # 3. Replace sequences of 2 or more newlines with " . " (space, period, space)
    # This is intended to create a sentence-like break for TTS.
    text = re.sub(r'\n{2,}', ' . ', text)

    # 4. Replace remaining single newlines with a single space
    text = text.replace('\n', ' ')

    # 5. Consolidate multiple spaces (including those formed from newline replacements) 
    # into a single space and strip leading/trailing whitespace.
    text = ' '.join(text.split()).strip()
    
    return text

def extract_chapters_from_epub(epub_file):
    try:
        book = epub.read_epub(epub_file)
    except ebooklib.epub.EpubException as e:
        print(f"Error: Failed to read EPUB file '{epub_file}'. Details: {e}")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading EPUB file '{epub_file}'. Details: {e}")
        return []

    chapters = []

    def process_toc_entry(toc_item, parent_title=None):
        # Case 1: toc_item is an actual link to content (ebooklib.epub.Link)
        if isinstance(toc_item, ebooklib.epub.Link):
            title = toc_item.title if toc_item.title else f"Chapter {len(chapters) + 1}"
            href = toc_item.href
            current_display_title = f"{parent_title} - {title}" if parent_title else title

            if not href or not isinstance(href, str):
                print(f"Warning: Skipping TOC Link '{title}' due to invalid or missing href: {href}")
                return

            file_href = href.split('#', 1)[0]
            fragment = href.split('#', 1)[1] if '#' in href else None

            if not file_href: # Handles hrefs that are only fragments (e.g., "#section1")
                print(f"Warning: Skipping TOC Link '{current_display_title}' with href '{href}' that points only to a fragment.")
                return

            item = book.get_item_with_href(file_href)
            if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                text_content = ""
                if fragment:
                    element = soup.find(id=fragment)
                    if element:
                        text_content = element.get_text(separator=' ')
                    else:
                        print(f"Warning: Fragment '{fragment}' not found in '{file_href}' for chapter '{current_display_title}'. Extracting full content of '{file_href}'.")
                        text_content = soup.get_text(separator=' ') # Fallback to full item content
                else:
                    text_content = soup.get_text(separator=' ')
                
                normalized_text = normalize_text(text_content)
                if normalized_text:
                    chapters.append({"title": current_display_title, "text": normalized_text})
                else:
                    print(f"Warning: No text extracted from '{current_display_title}' (href: {href}). Skipping.")
            else:
                print(f"Warning: Document item not found or not ITEM_DOCUMENT for href '{href}'. Skipping entry '{current_display_title}'.")

        # Case 2: toc_item is a list or tuple (often represents a section with sub-items or a list of links)
        elif isinstance(toc_item, (list, tuple)):
            # Check if it's a (Section, [sub_items]) structure
            if toc_item and len(toc_item) > 0 and isinstance(toc_item[0], ebooklib.epub.Section):
                section_obj = toc_item[0]
                section_title = section_obj.title if section_obj.title else "Unnamed Section"
                current_display_title = f"{parent_title} - {section_title}" if parent_title else section_title
                
                # sub_items are usually in the second element of the tuple if it's (Section, [items])
                sub_items = toc_item[1] if len(toc_item) > 1 and isinstance(toc_item[1], (list, tuple)) else []
                if not sub_items and hasattr(section_obj, 'children') and isinstance(section_obj.children, (list, tuple)):
                     # some epubs might have children directly on section object not in tuple structure
                    sub_items = section_obj.children

                for sub_nav_point in sub_items:
                    process_toc_entry(sub_nav_point, parent_title=current_display_title)
            else: # Otherwise, it's likely a flat list of items
                for sub_nav_point in toc_item:
                    process_toc_entry(sub_nav_point, parent_title=parent_title)

        # Case 3: toc_item is a Section (potentially a container without direct href or already handled)
        elif isinstance(toc_item, ebooklib.epub.Section):
            section_title = toc_item.title if toc_item.title else "Unnamed Section"
            current_display_title = f"{parent_title} - {section_title}" if parent_title else section_title
            # Sections might have 'children' that represent sub-TOC items (e.g. in NCX navmap)
            if hasattr(toc_item, 'children') and isinstance(toc_item.children, (list, tuple)):
                for child_item in toc_item.children:
                    process_toc_entry(child_item, parent_title=current_display_title)
            # If it has an href, it might be a direct link too (less common for pure sections)
            elif hasattr(toc_item, 'href') and toc_item.href:
                 # Create a Link-like object to process it as a chapter.
                 # This handles cases where a Section itself is a chapter link.
                pseudo_link = ebooklib.epub.Link(href=toc_item.href, title=section_title, uid=toc_item.uid if hasattr(toc_item, 'uid') else None)
                process_toc_entry(pseudo_link, parent_title=parent_title) # Use original parent_title here
            # else:
            #    print(f"Info: Encountered Section '{current_display_title}' without direct children or href to process further in this path.")

        # Case 4: Unknown entry type
        else:
            entry_repr = str(toc_item)[:100]
            print(f"Warning: Skipping unknown or unhandled TOC entry type: {type(toc_item)} - Content: '{entry_repr}...'.")

    # --- Main part of extract_chapters_from_epub ---
    if book.toc:
        print(f"Info: Processing TOC for '{epub_file}'. Found {len(book.toc)} top-level TOC items.")
        for toc_entry_item in book.toc: # toc_entry_item could be Link, Section, or tuple
            process_toc_entry(toc_entry_item) # Initial call, parent_title is None
        
        if chapters:
            print(f"Info: Successfully extracted {len(chapters)} chapters from TOC for '{epub_file}'.")
            return chapters
        else:
            print(f"Warning: No chapters were extracted from the TOC of '{epub_file}'. Proceeding to fallback.")
    else:
        print(f"Warning: No TOC found in EPUB file '{epub_file}'. Proceeding to fallback.")

    # Fallback: Extract all document items if TOC processing fails or yields no chapters
    print(f"Info: Falling back to extracting full document content for '{epub_file}'.")
    full_text = ""
    doc_item_count = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        doc_item_count += 1
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        text = soup.get_text(separator=' ')
        if text: # Append only if text exists
            full_text += text + " "
    
    if doc_item_count == 0:
        print(f"Warning: No document items (type ITEM_DOCUMENT) found in '{epub_file}' during fallback.")

    normalized_full_text = normalize_text(full_text)
    if normalized_full_text:
        print(f"Info: Extracted full document content as a single chapter for '{epub_file}'.")
        return [{"title": "Full Document", "text": normalized_full_text}]
    else:
        print(f"Warning: No text content found in '{epub_file}' after both TOC processing and fallback extraction.")
        return [{"title": "Empty Document", "text": ""}]

class PdfParser:
    """Parser for extracting chapters from PDF files."""
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        # Attempt to get absolute path for PyMuPDF if it's a relative path from Gradio file object
        if isinstance(self.pdf_file, str) and not os.path.isabs(self.pdf_file):
            # This assumes the file is accessible relative to the current working directory
            # For Gradio temp files, name attribute usually provides an absolute path
            pass


    def get_chapters(self):
        doc = None
        try:
            # print(f"Attempting to open PDF: {self.pdf_file}")
            doc = fitz.open(self.pdf_file)
        except FileNotFoundError:
            print(f"Error: PDF file not found at path: {self.pdf_file}")
            return [{'title': 'Error: PDF file not found', 'text': ''}]
        except fitz.PyMuPDFError as e: # More specific PyMuPDF error
            print(f"Error: PyMuPDFError occurred while opening PDF '{self.pdf_file}'. Details: {e}")
            return [{'title': 'Error: Could not open or read PDF (PyMuPDF specific error)', 'text': ''}]
        except Exception as e: # Catch other potential errors during open
            print(f"Error: An unexpected error occurred while opening PDF '{self.pdf_file}'. Details: {e}")
            return [{'title': 'Error: Could not open or read PDF', 'text': ''}]

        chapters = []
        toc = []
        try:
            toc = doc.get_toc()
        except Exception as e_toc:
            print(f"Warning: Could not retrieve TOC from PDF '{self.pdf_file}'. Error: {e_toc}. Proceeding with fallback.")
            toc = [] # Ensure toc is an empty list to trigger fallback

        if toc:
            print(f"Info: Processing TOC for PDF '{self.pdf_file}'. Found {len(toc)} entries.")
            for i, entry in enumerate(toc):
                level, title, page = entry
                if level != 1: # Process only top-level chapters
                    continue

                start_pdf_page = page - 1 # TOC pages are 1-based, PyMuPDF is 0-based
                end_pdf_page = doc.page_count # Default to end of document

                # Find the next entry at the same level (level 1) to determine the end page
                for next_i in range(i + 1, len(toc)):
                    next_level, _, next_page_num = toc[next_i]
                    if next_level == level:
                        end_pdf_page = next_page_num - 1
                        break
                
                # Ensure page numbers are valid
                if start_pdf_page < 0 or start_pdf_page >= doc.page_count:
                    print(f"Warning: Invalid start page ({page}) for chapter '{title}' in PDF '{self.pdf_file}'. Skipping.")
                    continue
                if end_pdf_page <= start_pdf_page:
                    # This can happen if next chapter starts on same page or due to TOC issues
                    # Set to read at least one page if start is valid.
                    end_pdf_page = start_pdf_page + 1 
                
                end_pdf_page = min(end_pdf_page, doc.page_count) # Cap at document page count

                chapter_text_parts = []
                for page_num_idx in range(start_pdf_page, end_pdf_page):
                    try:
                        page_obj = doc.load_page(page_num_idx)
                        chapter_text_parts.append(page_obj.get_text("text"))
                    except Exception as e_page:
                        print(f"Warning: Could not extract text from page {page_num_idx + 1} for chapter '{title}' in PDF '{self.pdf_file}'. Error: {e_page}")
                        chapter_text_parts.append("") # Append empty string for problematic page

                full_chapter_text = "".join(chapter_text_parts)
                # Normalization is done in process_file, but good to ensure non-empty here too
                if normalize_text(full_chapter_text): # only add if there's actual text
                    chapters.append({'title': title, 'text': full_chapter_text}) # Text will be normalized later
                else:
                    print(f"Info: Chapter '{title}' in PDF '{self.pdf_file}' resulted in no text content after extraction.")

        if chapters:
            print(f"Info: Successfully extracted {len(chapters)} chapters from PDF TOC for '{self.pdf_file}'.")
            doc.close()
            return chapters

        # Fallback: No chapters from TOC or TOC was empty/failed
        print(f"Info: No chapters extracted from PDF TOC (or TOC failed/empty) for '{self.pdf_file}'. Falling back to markdown conversion.")
        try:
            md_text = pymupdf4llm.to_markdown(doc) # Use the opened doc object
            # md_text = pymupdf4llm.to_markdown(self.pdf_file) # Alternative: pass path/object again
            if normalize_text(md_text):
                chapters.append({'title': 'Full Document', 'text': md_text}) # Text will be normalized later
            else:
                print(f"Info: Markdown fallback for PDF '{self.pdf_file}' resulted in no text content.")
                chapters.append({'title': 'Empty Document', 'text': ''})
        except Exception as e_md:
            print(f"Error: Failed to convert PDF '{self.pdf_file}' to markdown. Error: {e_md}")
            chapters.append({'title': 'Full Document (Markdown Extraction Failed)', 'text': ''})
        
        doc.close()
        # If chapters is still empty (e.g. markdown failed and TOC yielded nothing), this will return the error from markdown
        # or an empty list if markdown produced nothing.
        # Ensure consistent return of list of dicts.
        if not chapters: # Should ideally be caught by logic above, but as a safeguard
             print(f"Warning: All extraction methods for PDF '{self.pdf_file}' yielded no chapters. Returning Empty Document.")
             return [{'title': 'Empty Document', 'text': ''}]
        return chapters

def generate_first(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    text = normalize_text(text) if CHAR_LIMIT is None else normalize_text(text.strip()[:CHAR_LIMIT])
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[min(len(ps)-1, 509)]
        try:
            audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU.')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        return (24000, audio.numpy()), ps
    return None, ''

def predict(text, voice='af_heart', speed=1):
    return generate_first(text, voice, speed, use_gpu=False)[0]

def tokenize_first(text, voice='af_heart'):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, stream_all=False, document=None, chapter_title=None):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE

    # If stream_all is False, just stream the current chapter
    if not stream_all or not document or not chapter_title:
        text = normalize_text(text) if CHAR_LIMIT is None else normalize_text(text.strip()[:CHAR_LIMIT])
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[min(len(ps)-1, 509)]
            try:
                audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info('Switching to CPU')
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()
        return

    # If stream_all is True, stream the selected chapter and all subsequent chapters
    with open(os.path.join("processed_documents", document), "r") as f:
        data = json.load(f)
    chapters = data["chapters"]
    start_index = next(i for i, chapter in enumerate(chapters) if chapter["title"] == chapter_title)
    
    for chapter in chapters[start_index:]:
        text = normalize_text(chapter["text"]) if CHAR_LIMIT is None else normalize_text(chapter["text"].strip()[:CHAR_LIMIT])
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[min(len(ps)-1, 509)]
            try:
                audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info('Switching to CPU')
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()

def process_file_with_timestamps(file, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    text = normalize_text(process_file(file.name))
    text = text if CHAR_LIMIT is None else text[:CHAR_LIMIT]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    chunks = []
    html = '<table><tr><th>Start</th><th>End</th><th>Text</th></tr>'
    for gs, ps, tks in pipeline(text, voice, speed, model=models[use_gpu]):
        ref_s = pack[min(len(ps)-1, 509)]
        audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
        chunks.append(audio.numpy())
        if tks and tks[0].start_ts is not None:
            for t in tks:
                if t.start_ts is None or t.end_ts is None:
                    continue
                html += f'<tr><td>{t.start_ts:.2f}</td><td>{t.end_ts:.2f}</td><td>{t.text}</td></tr>'
    html += '</table>'
    return chunks, html

def stream_file(file, voice, speed, use_gpu):
    chunks, html = process_file_with_timestamps(file, voice, speed, use_gpu)
    return chunks, html

def stream_audio_from_chunks(chunks, streaming_active):
    for rate, audio in chunks:
        if not streaming_active:
            break
        yield rate, audio

def stream_file_from_chunk(chunk_state, chunk_data):
    start_chunk = chunk_data['chunk'] if chunk_data and 'chunk' in chunk_data else 0
    for rate, audio in chunk_state[start_chunk:]:
        yield rate, audio

with open('en.txt', 'r') as r:
    random_quotes = [line.strip() for line in r]

def get_random_quote():
    return random.choice(random_quotes)

def get_gatsby():
    with open('gatsby5k.md', 'r') as r:
        return r.read().strip()

def get_frankenstein():
    with open('frankenstein5k.md', 'r') as r:
        return r.read().strip()

CHOICES = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
}
for v in CHOICES.values():
    pipelines[v[0]].load_voice(v)

TOKEN_NOTE = '''
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`
'''

STREAM_NOTE = '\n\n'.join(['⚠️ Gradio bug might yield no audio on first Stream click'] + 
                         ([f'✂️ Capped at {CHAR_LIMIT} characters'] if CHAR_LIMIT else []))

if not os.path.exists("processed_documents"):
    os.makedirs("processed_documents")

def process_file(file):
    """Process uploaded files (EPUB, PDF, TXT) and return chapters or full text."""
    file_path = file.name if hasattr(file, 'name') else file
    if file_path.endswith('.epub'):
        return extract_chapters_from_epub(file_path)
    elif file_path.endswith('.pdf'):
        parser = PdfParser(file_path)
        chapters = parser.get_chapters()
        for chapter in chapters:
            chapter['text'] = normalize_text(chapter['text'])
        return chapters
    else: # Handles TXT files and any other unspecified file types
        text = ""
        try:
            try:
                # Attempt to read with UTF-8 first
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                # print(f"Info: Successfully read TXT file '{file_path}' with UTF-8 encoding.")
            except UnicodeDecodeError:
                print(f"Warning: Could not decode TXT file '{file_path}' with UTF-8. Attempting fallback to latin-1.")
                try:
                    # Fallback to latin-1 if UTF-8 fails
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text = f.read()
                    print(f"Info: Successfully read TXT file '{file_path}' with latin-1 encoding.")
                except UnicodeDecodeError as e_latin1:
                    print(f"Error: Failed to decode TXT file '{file_path}' with both UTF-8 and latin-1 encodings. Error: {e_latin1}")
                    return [{'title': 'Error: TXT file encoding issue (tried UTF-8, latin-1)', 'text': ''}]
                except Exception as e_fallback_read: # Other errors during fallback read
                    print(f"Error: Could not read TXT file '{file_path}' with fallback encoding latin-1. Error: {e_fallback_read}")
                    return [{'title': 'Error: Could not read TXT file with fallback encoding', 'text': ''}]
            
            normalized_text = normalize_text(text)
            return [{'title': 'Full Document', 'text': normalized_text}]

        except FileNotFoundError:
            print(f"Error: TXT file not found at path: {file_path}")
            return [{'title': 'Error: TXT file not found', 'text': ''}]
        except IOError as e_io:
            print(f"Error: IOError occurred while reading TXT file '{file_path}'. Details: {e_io}")
            return [{'title': 'Error: Could not read TXT file (IO error)', 'text': ''}]
        except Exception as e_general: # Catch-all for any other unexpected errors
            print(f"Error: An unexpected error occurred while processing TXT file '{file_path}'. Details: {e_general}")
            return [{'title': 'Error: Unexpected error processing TXT file', 'text': ''}]

def generate_unique_name(original_name):
    base_name = os.path.splitext(os.path.basename(original_name))[0]
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}.json"

def save_chapters(chapters, unique_name):
    data = {"document_name": unique_name, "chapters": chapters}
    with open(os.path.join("processed_documents", unique_name), "w") as f:
        json.dump(data, f)

def get_documents():
    return [f for f in os.listdir("processed_documents") if f.endswith('.json')]

def get_chapters(document):
    if not document:
        return gr.update(choices=[], value=None)
    with open(os.path.join("processed_documents", document), "r") as f:
        data = json.load(f)
    chapters = [chapter["title"] for chapter in data["chapters"]]
    return gr.update(choices=chapters, value=chapters[0] if chapters else None)

def load_chapter_text(document, chapter_title):
    if not document or not chapter_title:
        return ""
    with open(os.path.join("processed_documents", document), "r") as f:
        data = json.load(f)
    for chapter in data["chapters"]:
        if chapter["title"] == chapter_title:
            return chapter["text"]
    return ""

def process_and_save(file):
    if file is None:
        # Use gr.update() for all outputs if only a status message is intended for the third output
        # Or, if the first two outputs are dropdowns that shouldn't change, explicitly use gr.update() for them.
        # Assuming the first two are dropdowns and status is the third:
        return gr.update(), gr.update(), "Please upload a file."

    chapters = process_file(file)

    # Check for error condition returned by process_file
    # An error is indicated if `chapters` is a list containing exactly one dictionary, 
    # and that dictionary's 'title' key starts with "Error:"
    is_error = False
    error_message = ""
    if isinstance(chapters, list) and len(chapters) == 1:
        first_item = chapters[0]
        if isinstance(first_item, dict) and first_item.get('title', '').startswith('Error:'):
            is_error = True
            error_message = first_item['title']

    if is_error:
        print(f"Info: File processing failed. Error: {error_message}")
        # Refresh document list (without selecting the failed one) and clear chapter list
        # Return the error message as the status
        return (
            gr.update(choices=get_documents(), value=None), # Update document_dropdown
            gr.update(choices=[], value=None),              # Update chapter_dropdown
            error_message                                   # Update status textbox
        )
    else:
        # Proceed with saving and updating UI for successful processing
        # (This includes cases like "Empty Document" or "Full Document" which are not errors)
        unique_name = generate_unique_name(file.name)
        save_chapters(chapters, unique_name)
        
        documents = get_documents() # Re-fetch to include the new document
        chapter_titles = [chapter["title"] for chapter in chapters if isinstance(chapter, dict) and "title" in chapter]
        
        success_message = f"Document '{os.path.basename(unique_name)}' processed and saved."
        if not chapter_titles: # Handles "Empty Document" or if chapters list was empty for some reason
             print(f"Info: Document '{unique_name}' processed, but no chapter titles found to select. Chapters: {chapters}")
             # Select the processed document, but no specific chapter
             return (
                gr.update(choices=documents, value=unique_name),
                gr.update(choices=[], value=None),
                success_message
            )

        return (
            gr.update(choices=documents, value=unique_name),
            gr.update(choices=chapter_titles, value=chapter_titles[0]), # Select the first chapter by default
            success_message
        )
    
# Python function to handle stop button click
def handle_stop_button():
    print("DEBUG: Stop button Python handler: Setting streaming_active to False and out_stream to None.")
    return (False, None) # Return False for streaming_active, None for out_stream
    
def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, streaming_active=True, stream_all=False, document=None, chapter_title=None):
    print(f"DEBUG: generate_all called. Initial streaming_active: {streaming_active}, stream_all: {stream_all}, document: {document}, chapter_title: {chapter_title}")
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE

    # If stream_all is False, just stream the current chapter (original behavior)
    if not stream_all or not document or not chapter_title:
        text = normalize_text(text) if CHAR_LIMIT is None else normalize_text(text.strip()[:CHAR_LIMIT])
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            print(f"DEBUG: generate_all (single chunk loop): streaming_active={streaming_active}")
            if not streaming_active:
                print(f"DEBUG: generate_all (single chunk loop): streaming_active is False, breaking.")
                break
            ref_s = pack[min(len(ps)-1, 509)]
            try:
                audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info('Switching to CPU')
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()
        print("DEBUG: generate_all exiting (after single chunk loop or if not stream_all).")
        return

    # If stream_all is True, stream the selected chapter and all subsequent chapters
    print(f"DEBUG: generate_all: stream_all is True. Document: {document}, Chapter: {chapter_title}")
    with open(os.path.join("processed_documents", document), "r") as f:
        data = json.load(f)
    chapters = data["chapters"]
    # Find the index of the selected chapter
    start_index = next(i for i, chapter in enumerate(chapters) if chapter["title"] == chapter_title)
    
    # Stream each chapter from the selected one onward
    for idx, chapter in enumerate(chapters[start_index:]):
        print(f"DEBUG: generate_all (chapter loop): streaming_active={streaming_active} for chapter '{chapter.get('title', 'N/A')}' (Index {start_index + idx})")
        if not streaming_active:
            print(f"DEBUG: generate_all (chapter loop): streaming_active is False, breaking from chapter loop.")
            break
        text = normalize_text(chapter["text"]) if CHAR_LIMIT is None else normalize_text(chapter["text"].strip()[:CHAR_LIMIT])
        first = True
        for _, ps, _ in pipeline(text, voice, speed):
            print(f"DEBUG: generate_all (multi-chapter chunk loop): streaming_active={streaming_active}")
            if not streaming_active:
                print(f"DEBUG: generate_all (multi-chapter chunk loop): streaming_active is False, breaking from chunk loop.")
                break
            ref_s = pack[min(len(ps)-1, 509)]
            try:
                audio = forward_gpu(ps, ref_s, speed) if use_gpu else models[False](ps, ref_s, speed)
            except gr.exceptions.Error as e:
                if use_gpu:
                    gr.Warning(str(e))
                    gr.Info('Switching to CPU')
                    audio = models[False](ps, ref_s, speed)
                else:
                    raise gr.Error(e)
            yield 24000, audio.numpy()
            if first:
                first = False
                yield 24000, torch.zeros(1).numpy()
    print("DEBUG: generate_all exiting (after stream_all chapter loop or break).")            
                

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    with gr.Accordion('Output Tokens', open=True):
        out_ps = gr.Textbox(interactive=False, show_label=False)
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        gr.Markdown(TOKEN_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
        # stream_file_btn = gr.Button('Stream File', variant='primary') # Commented out
    # process_btn, file_upload, and status are moved to the first column.
    # Markdown related to process_btn is also moved.
    # file_viewer = gr.HTML(label="File Viewer") # Commented out
    # chunk_state = gr.State(value=[]) # Commented out
    streaming_active = gr.State(value=False) # Keep: Used by stream_btn
    gr.HTML("""
    <script>
    function stopAudio() {
        console.log("DEBUG: stopAudio JS: function called");
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(audio => {
            console.log("DEBUG: stopAudio JS: Pausing audio element:", audio);
            audio.pause();
            audio.currentTime = 0;
        });
    }
    window.addEventListener('message', (event) => {
        if (event.data.chunk !== undefined) {
            document.getElementById('chunk_input').value = JSON.stringify({chunk: event.data.chunk});
            document.getElementById('chunk_trigger').click();
        }
    });
    </script>
    """)
    # chunk_input = gr.JSON(visible=False, elem_id="chunk_input") # Commented out
    # chunk_trigger = gr.Button("Trigger Chunk", visible=False, elem_id="chunk_trigger") # Commented out

BANNER_TEXT = '''
# Kokoro-Plus [(getgoingfast.pro)](https://www.getgoingfast.pro)

[***Kokoro*** **is an open-weight TTS model with 82 million parameters.**](https://huggingface.co/hexgrad/Kokoro-82M)  

[Listen to good music!](https://music.youtube.com/channel/UCY658vbL6S2zlRomNHoX54Q)
'''

API_OPEN = os.getenv('SPACE_ID') != 'hexgrad/Kokoro-TTS'



with gr.Blocks() as app:
    gr.Markdown(BANNER_TEXT)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload and Process New Document")
            file_upload = gr.File(label="Upload EPUB/PDF/TXT")
            process_btn = gr.Button('Process and Save Chapters', variant='secondary')
            status = gr.Textbox(label="Status", interactive=False, placeholder="Upload status will appear here...")
            gr.Markdown("---")  # Visual separator
            
            gr.Markdown("### Load Pre-processed Document")
            with gr.Row():
                document_dropdown = gr.Dropdown(
                    label="Select Document",
                    choices=[""] + get_documents(),  # Blank default option
                    value=""  # Pre-select the blank option
                )
                chapter_dropdown = gr.Dropdown(
                    label="Select Chapter",
                    choices=[],
                    value=None
                )
            text = gr.Textbox(label='Input Text')
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice')
                use_gpu = gr.Dropdown([('ZeroGPU 🚀', True), ('CPU 🐌', False)], value=CUDA_AVAILABLE, label='Hardware')
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            stream_all_checkbox = gr.Checkbox(label="Stream All Chapters", value=False)
            random_btn = gr.Button('🎲 Random Quote 💬', variant='secondary')
            gatsby_btn = gr.Button('🥂 Gatsby 📕', variant='secondary')
            frankenstein_btn = gr.Button('💀 Frankenstein 📗', variant='secondary')
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ['Generate', 'Stream'])

    # Event handlers
    document_dropdown.change(fn=get_chapters, inputs=[document_dropdown], outputs=[chapter_dropdown])
    chapter_dropdown.change(fn=load_chapter_text, inputs=[document_dropdown, chapter_dropdown], outputs=[text])
    process_btn.click(fn=process_and_save, inputs=[file_upload], outputs=[document_dropdown, chapter_dropdown, status])
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])
    generate_btn.click(fn=generate_first, inputs=[text, voice, speed, use_gpu], outputs=[out_audio, out_ps])
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])

    stream_event = stream_btn.click(
        fn=lambda: (True, None),
        outputs=[streaming_active, out_stream]
    ).then(
        fn=generate_all,
        inputs=[text, voice, speed, use_gpu, streaming_active, stream_all_checkbox, document_dropdown, chapter_dropdown],
        outputs=[out_stream]
    )

    # file_stream_event = stream_file_btn.click(
    #     fn=lambda: (True, None),
    #     outputs=[streaming_active, out_stream]
    # ).then(
    #     fn=stream_file,
    #     inputs=[file_upload, voice, speed, use_gpu],
    #     outputs=[chunk_state, file_viewer]
    # ).then(
    #     fn=stream_audio_from_chunks,
    #     inputs=[chunk_state, streaming_active],
    #     outputs=[out_stream]
    # )

    stop_btn.click(
        fn=handle_stop_button, 
        outputs=[streaming_active, out_stream], # Update outputs to include out_stream
        js="stopAudio", 
        cancels=[stream_event] 
    )

    # chunk_trigger.click(fn=stream_file_from_chunk, inputs=[chunk_state, chunk_input], outputs=[out_stream]) # Commented out

if __name__ == '__main__':
    app.queue(api_open=API_OPEN).launch(share=API_OPEN)
