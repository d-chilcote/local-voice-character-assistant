---
name: memory_erasure
description: Allows the agent to delete its memory and conversation history.
---

# Memory Erasure Skill

This skill allows the agent to wipe its persistent memory file and clear its current conversation history.
It should be triggered when the user explicitly asks to "erase memory", "wipe memory", or says "you have amnesia".

## Capabilities
- Deletes the JSON memory file (if it exists).
- Clears the active conversation history list.

## Usage
- **Trigger**: "Erase your memory", "Forget everything", "You have amnesia".
- **Action**: `memory_erasure`

## Parameters
- None required (history and memory_file are injected automatically).
