from playwright.sync_api import Page, expect

def test_ui_changes(page: Page):
    """
    This test verifies the UI changes made to the Gradio application.
    It checks that the file upload component is present in the main UI,
    and that the "Process and Save" button has been removed from the "Stream" tab.
    """
    # 1. Arrange: Go to the application's URL.
    # The user will need to replace this with the correct URL.
    page.goto("http://127.0.0.1:7860")

    # 2. Assert: Check that the file upload component is visible in the main UI.
    expect(page.get_by_label("Upload EPUB/PDF/TXT")).to_be_visible()

    # 3. Act: Click on the "Stream" tab.
    page.get_by_role("tab", name="Stream").click()

    # 4. Assert: Check that the "Process and Save Chapters" button is not visible.
    expect(page.get_by_role("button", name="Process and Save Chapters")).not_to_be_visible()

    # 5. Screenshot: Capture the final result for visual verification.
    page.screenshot(path="jules-scratch/verification/verification.png")