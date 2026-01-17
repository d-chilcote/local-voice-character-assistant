---
name: google_search
description: Performs a grounded Google Search via Gemini 2.5 Flash-Lite to find factual information, news, and statistics.
license: Apache-2.0
---

# Google Search Skill

This skill provides the agent with the ability to perform live Google Searches through the Gemini 2.5 API. Use this whenever you need to verify facts, find recent news, or retrieve specific data points that are not present in your local knowledge base.

## 🛠️ Usage Guidelines

1.  **When to Use**: 
    - Questions about current events (e.g., "What happened in the news today?").
    - Verifying specific facts (e.g., "What is the population of Tokyo in 2024?").
    - Getting real-time data (e.g., stock prices, weather, sports scores).
2.  **How to Call**:
    - Select the `google_search` skill.
    - Provide a clear, concise `query` string.
3.  **Synthesizing Results**:
    - The skill returns grounded results from Google Search. 
    - Always attribute the information if possible.
    - If the search fails or returns no relevant data, inform the user honestly.

## 📝 Examples

- **User**: "Who is the current Prime Minister of the UK?"
  **Action**: `google_search`
  **Arguments**: `{"query": "current Prime Minister of the UK"}`

- **User**: "What was the score of the Lakers game last night?"
  **Action**: `google_search`
  **Arguments**: `{"query": "Lakers game score last night"}`

## 📜 Metadata
| Field | Value |
|-------|-------|
| Name  | google_search |
| API   | Gemini 2.5 Flash-Lite |
| Tools | Google Search Grounding |
