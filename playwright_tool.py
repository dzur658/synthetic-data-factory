import asyncio
import httpx  # Using httpx for async requests
import sys
from playwright.async_api import async_playwright
from unstructured.partition.html import partition_html
from unstructured.documents.elements import Title
from unstructured.documents.elements import NarrativeText, Text
from unstructured.staging.base import elements_to_json

# --- Configuration ---
# !!! IMPORTANT: Change this to your SearXNG instance URL
SEARXNG_URL = "http://127.0.0.1:8081/search" 
MAX_RESULTS_TO_PARSE = 3 # How many top results to parse per query

def add_markdonwn(text: str, element_class: str) -> str:
    """
    Adds markdown formatting based on the element class.
    """
    if element_class == "Title":
        return f"# {text}"
    elif element_class == "Text":
        return f"*{text}*"
    else:
        # NarrativeText and all others as standard text
        return text

# --- 2. The Python Functions that Implement the Tool ---

async def get_and_parse_page(url: str) -> tuple[str, str]:
    """
    Uses Playwright to browse to a URL and unstructured to parse its content.
    Returns a tuple of (clean_text, url).
    """
    # Using stderr for progress logs so stdout can be clean for piping
    print(f"  > Parsing: {url}", file=sys.stderr)
    try:
        async with async_playwright() as p:
            # We use Chromium. You must run `playwright install chromium` first.
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, timeout=10000)
            
            # Get the fully rendered HTML
            html_content = await page.content()
            await browser.close()

            # Use 'unstructured' to parse the HTML
            # This is "schema-less" - it finds the main content automatically.
            elements = partition_html(text=html_content)
            
            # Filter for just the useful text elements
            clean_text_parts = []
            for e in elements:
                # Title, NarrativeText, and Text are good general-purpose elements
                if isinstance(e, (Title, NarrativeText, Text)):
                    clean_text_part = add_markdonwn(e.text, e.__class__.__name__)

                    clean_text_parts.append(clean_text_part)

            clean_text = "\n".join(clean_text_parts)
            
            if not clean_text:
                print(f"  > No text found at: {url}", file=sys.stderr)
                return None, url

            return clean_text, url
            
    except Exception as e:
        print(f"  > Error parsing {url}: {e}", file=sys.stderr)
        return None, url

async def search_and_parse_web(query: str) -> str:
    """
    This is the main function that implements the tool.
    It chains SearXNG, Playwright, and Unstructured.
    This function is designed to be imported by other scripts.
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
            response.raise_for_status() # Raise error for bad responses
            results = response.json()
            
            # Extract URLs from the results
            urls = [r["url"] for r in results.get("results", [])]
            if not urls:
                return "No search results found."

            # Limit to top N results
            urls = urls[:MAX_RESULTS_TO_PARSE]
                
            print(f"--- 2. Found {len(urls)} URLs. Starting parser... ---", file=sys.stderr)

    except Exception as e:
        print(f"Error querying SearXNG: {e}", file=sys.stderr)
        return f"Error querying SearXNG: {e}"

    # --- Step 2 & 3: Browse and Parse all pages concurrently ---
    tasks = [get_and_parse_page(url) for url in urls]
    parsed_results = await asyncio.gather(*tasks)

    # --- Step 4: Format the output for the LLM ---
    final_output = []
    for text, url in parsed_results:
        if text:
            # Format as context for the LLM
            final_output.append(f"[Source: {url}]\n{text}\n\n" + "-"*20 + "\n")
    
    print(f"--- 3. Parsing complete. Returning content. ---", file=sys.stderr)
    # Return a single string of all context
    return "\n".join(final_output)

if __name__ == "__main__":
    # This allows you to test the tool directly from the command line
    # Usage: python tool_prototype.py "your search query"
    if len(sys.argv) < 2:
        print("Usage: python tool_prototype.py \"<search query>\"", file=sys.stderr)
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    
    print(f"Testing search_and_parse_web with query: '{query}'", file=sys.stderr)
    
    async def test_run():
        # The script will now print the *clean output* to stdout
        # and logs to stderr.
        output = await search_and_parse_web(query)
        print(output) # This prints the final result to stdout

        with open("results.md", "w", encoding="utf-8") as f:
            f.write(output)

    asyncio.run(test_run())