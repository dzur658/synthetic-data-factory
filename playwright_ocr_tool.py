import asyncio
from xml.dom.minidom import Text
import httpx  # Using httpx for async requests
import sys
import base64
from playwright.async_api import async_playwright
# We are bringing unstructured back to create a hybrid parser
from unstructured.partition.html import partition_html
from unstructured.documents.elements import Title, NarrativeText, Text

import vlm_ollama

# --- Configuration ---
SEARXNG_URL = "http://127.0.0.1:8081/search"
MAX_RESULTS_TO_PARSE = 3  # 1 result is good for this complex hybrid parsing
# I'll update this to a newer VLM, as we discussed.
# Make sure you do: ollama pull qwen3-vl:8b
VLM_MODEL = "qwen3-vl:4b-instruct-bf16" 

# --- 2. The Python Functions that Implement the Tool ---

def add_markdonwn(text: str, element_class: str) -> str:
    """
    Adds markdown formatting based on the element class.
    """
    if element_class == "Title":
        return f"## {text}"
    elif element_class == "Text":
        return f"*{text}*"
    else:
        # NarrativeText and all others as standard text
        return text

async def get_hybrid_parsed_page(url: str) -> tuple[str, str]:
    """
    Uses a hybrid approach:
    1. unstructured parses all text elements from HTML into Markdown.
    2. A VLM (like Qwen-VL) parses all visual elements (charts, images) from a screenshot.
    Returns a tuple of (combined_markdown, url).
    """
    print(f"  > Hybrid parsing: {url}", file=sys.stderr)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, timeout=10000)
            
            # 1. Get BOTH HTML content and a screenshot
            html_content = await page.content()
            screenshot_bytes = await page.screenshot(full_page=True)
            
            await browser.close()
            
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        # 2. Define our two parsing tasks to run concurrently
        
        async def parse_text_with_unstructured():
            """Parses HTML text into Markdown."""
            print(f"    > Running unstructured text parser...", file=sys.stderr)
            try:
                elements = partition_html(text=html_content)

                clean_text_parts = []

                for e in elements:
                    # Title, NarrativeText, and Text are good general-purpose elements
                    if isinstance(e, (Title, NarrativeText, Text)):
                        clean_text_parts.append(add_markdonwn(e.text, element_class=e.__class__.__name__))

                markdown = "\n".join(clean_text_parts)
                
                return markdown
            except Exception as e:
                print(f"    > Unstructured parser failed: {e}", file=sys.stderr)
                return "[Unstructured text parsing failed]"

        async def parse_images_with_vlm():
            """Sends screenshot to VLM to describe *only* visual elements."""
            print(f"    > Running {VLM_MODEL} vision parser...", file=sys.stderr)
            
            # This prompt is crucial. It tells the VLM to *ignore* the text
            # that unstructured is already handling.
            vlm_prompt = (
                "Analyze this webpage screenshot. "
                "Your task is to describe only the visual elements like charts, graphs, and important photographs. "
                "Do NOT extract or repeat the main body text, paragraphs, or headers."
            )
            
            try:
                response = await asyncio.to_thread(
                    vlm_ollama.interpret_page,
                    VLM_MODEL,
                    screenshot_base64,
                    vlm_prompt,
                    0.4
                )
                return response
            except Exception as e:
                print(f"    > VLM parser failed: {e}")
                return f"[VLM analysis failed: {e}]"

        # 3. Run both parsers in parallel
        text_markdown, visual_descriptions = await asyncio.gather(
            parse_text_with_unstructured(),
            parse_images_with_vlm()
        )

        # 4. Combine the results into one document
        final_markdown = (
            f"{text_markdown}\n\n"
            "---\n\n"
            "# Visual Elements Analysis (from VLM)\n\n"
            f"{visual_descriptions}"
        )

        return final_markdown, url
            
    except Exception as e:
        print(f"  > Error parsing {url}: {e}", file=sys.stderr)
        return None, url

async def search_and_parse_web(query: str) -> str:
    """
    This is the main function that implements the tool.
    It chains SearXNG, Playwright, and the hybrid parser.
    """
    print(f"\n--- 1. Searching SearXNG for: '{query}' ---", file=sys.stderr)
    
    # --- Step 1: Query SearXNG ---
    urls = []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                SEARXNG_URL,
                params={"q": query, "format": "json"},
                headers={"Accept": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            results = response.json()
            
            urls = [r["url"] for r in results.get("results", [])]

            # Truncate Urls to MAX_RESULTS_TO_PARSE
            urls = urls[:MAX_RESULTS_TO_PARSE]

            if not urls:
                return "No search results found."
                
            print(f"--- 2. Found {len(urls)} URLs. Starting hybrid parser... ---", file=sys.stderr)

    except Exception as e:
        print(f"Error querying SearXNG: {e}", file=sys.stderr)
        return f"Error querying SearXNG: {e}"

    # --- Step 2 & 3: Browse and Parse all pages concurrently ---
    tasks = [get_hybrid_parsed_page(url) for url in urls]
    parsed_results = await asyncio.gather(*tasks)

    # --- Step 4: Format the output for the LLM ---
    final_output = []
    for text, url in parsed_results:
        if text:
            final_output.append(f"[Source: {url}]\n #Webpage Title: {text}\n\n" + "-"*20 + "\n")
    
    print(f"--- 3. Parsing complete. Returning content. ---", file=sys.stderr)
    return "\n".join(final_output)


# --- 3. The Main Orchestrator ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tool_prototype.py \"<search query>\"", file=sys.stderr)
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    
    print(f"Testing search_and_parse_web with query: '{query}'", file=sys.stderr)
    
    async def test_run():
        output = await search_and_parse_web(query)
        print(output) # This prints the final result to stdout

        with open("tool_output.md", "a", encoding="utf-8") as f:
            f.write(output)

    asyncio.run(test_run())

