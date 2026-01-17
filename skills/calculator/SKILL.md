---
name: calculator
description: Performs safe mathematical calculations using Python.
license: Apache-2.0
---

# Calculator Skill

Use this skill for any math question that is too complex for mental arithmetic or requires precision.

## 🛠️ Usage Guidelines

1.  **When to Use**:
    - "What is the square root of 144?"
    - "Calculate 15% of 850."
    - "Convert 30 Celsius to Fahrenheit."
2.  **Arguments**:
    - `expression`: A valid Python mathematical expression string.
    - **Note**: The expression is sanitized and evaluated safely.

## 📝 Examples

- **User**: "What is 234 times 56?"
  **Action**: `calculator`
  **Arguments**: `{"expression": "234 * 56"}`

## 📜 Metadata
| Field | Value |
|-------|-------|
| Name  | calculator |
| Type  | Logic / Math |
