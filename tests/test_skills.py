import pytest
import os
import shutil
from unittest.mock import MagicMock, patch
from skills.registry import SkillRegistry

@pytest.fixture
def mock_skills_dir(tmp_path):
    """Creates a temporary skills directory with a mock skill."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    
    skill_dir = skills_dir / "test_skill"
    skill_dir.mkdir()
    
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: test_skill
description: A test skill
---
Instructions for test skill.""")
    
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script_py = scripts_dir / "main.py"
    script_py.write_text("""
def execute(query: str) -> str:
    return f"Executed with {query}"
""")
    
    return str(skills_dir)

def test_skill_discovery(mock_skills_dir):
    """Test that skills are discovered correctly."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    assert "test_skill" in registry._skills
    assert registry._skills["test_skill"].description == "A test skill"

def test_skill_instructions(mock_skills_dir):
    """Test that instructions are formatted correctly for the prompt."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    instructions = registry.get_skill_instructions()
    assert "AVAILABLE SKILLS:" in instructions
    assert "--- SKILL: test_skill ---" in instructions
    assert "Instructions for test skill." in instructions

def test_skill_execution(mock_skills_dir):
    """Test that a skill script can be executed."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    result = registry.execute_skill("test_skill", query="hello")
    assert result == "Executed with hello"

def test_google_search_skill_real():
    """Test the real google_search skill discovery if it exists."""
    from skills.registry import registry as real_registry
    # Force rediscovery to pick up new files
    real_registry.discover_skills()
    
    expected_skills = ["google_search", "system_info", "calculator", "todo_list"]
    for skill in expected_skills:
        assert skill in real_registry._skills, f"Skill {skill} not found in registry"
