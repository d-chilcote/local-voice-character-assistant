---
name: todo_list
description: Manages a simple todo list file for the user.
license: Apache-2.0
---

# Todo List Skill

Allows the agent to read from and write to a persistent TODO list file.

## 🛠️ Usage Guidelines

1.  **Actions**:
    - `add`: Append a new item.
    - `read`: List all items.
    - `clear`: Wipe the list.
2.  **Arguments**:
    - `action`: "add" | "read" | "clear"
    - `item`: The text to add (required for 'add' action).

## 📝 Examples

- **User**: "Remind me to buy milk."
  **Action**: `todo_list`
  **Arguments**: `{"action": "add", "item": "Buy milk"}`

- **User**: "What do I need to do?"
  **Action**: `todo_list`
  **Arguments**: `{"action": "read"}`

## 📜 Metadata
| Field | Value |
|-------|-------|
| Name  | todo_list |
| Type  | Persistence / File |
