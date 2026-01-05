from ddgs import DDGS
import sys

print(f"Testing DuckDuckGo Search...")
query = "Super Bowl winner 2024"

try:
    results = list(DDGS().text(query, max_results=3))
    print(f"Results Found: {len(results)}")
    for r in results:
        print(f"- {r['title']}")
except Exception as e:
    print(f"Error: {e}")
