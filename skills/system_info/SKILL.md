---
name: system_info
description: Retrieves diagnostic information about the local system (OS, CPU, Memory).
license: Apache-2.0
---

# System Info Skill

Use this skill when the user asks about the computer's status, hardware specifications, or current resource usage.

## 🛠️ Usage Guidelines

1.  **When to Use**:
    - "How much RAM do I have?"
    - "What OS version is this?"
    - "Is my CPU running hot?" (Returns usage %)
2.  **Arguments**:
    - None required. Call with empty arguments.

## 📝 Examples

- **User**: "System status report."
  **Action**: `system_info`
  **Arguments**: `{}`

## 📜 Metadata
| Field | Value |
|-------|-------|
| Name  | system_info |
| Type  | Local / Diagnostic |
