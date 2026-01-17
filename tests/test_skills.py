import pytest
import os
import json
from unittest.mock import MagicMock, patch
from skills.registry import SkillRegistry, registry as real_registry

@pytest.fixture
def mock_skills_dir(tmp_path):
    """Creates a temporary skills directory with various test scenarios."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    
    # 1. Standard Skill
    skill_dir = skills_dir / "test_skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: test_skill\ndescription: A test skill\n---\n")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "main.py").write_text("def execute(query: str) -> str: return f'Executed with {query}'")

    # 2. Skill with Missing Execute
    bad_skill_dir = skills_dir / "bad_skill"
    bad_skill_dir.mkdir()
    (bad_skill_dir / "SKILL.md").write_text("---\nname: bad_skill\n---\n")
    (bad_skill_dir / "scripts").mkdir()
    (bad_skill_dir / "scripts" / "main.py").write_text("def not_execute(): pass")

    # 3. Skill with no python script
    empty_skill_dir = skills_dir / "empty_skill"
    empty_skill_dir.mkdir()
    (empty_skill_dir / "SKILL.md").write_text("---\nname: empty_skill\n---\n")
    (empty_skill_dir / "scripts").mkdir()

    return str(skills_dir)

def test_skill_discovery(mock_skills_dir):
    """Test that skills are discovered correctly."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    assert "test_skill" in registry._skills
    assert "bad_skill" in registry._skills
    assert registry._skills["test_skill"].description == "A test skill"

def test_skill_instructions(mock_skills_dir):
    """Test that instructions are formatted correctly for the prompt."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    instructions = registry.get_skill_instructions()
    assert "AVAILABLE SKILLS:" in instructions
    assert "--- SKILL: test_skill ---" in instructions

def test_skill_execution(mock_skills_dir):
    """Test that a skill script can be executed."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    result = registry.execute_skill("test_skill", query="hello")
    assert result == "Executed with hello"

def test_skill_argument_filtering(mock_skills_dir):
    """Test that extra arguments are filtered out based on signature."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    # 'test_skill' only takes 'query'
    result = registry.execute_skill("test_skill", query="hello", extra="ignore me")
    assert result == "Executed with hello"

def test_skill_missing_execute(mock_skills_dir):
    """Test handling of script with missing execute function."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    result = registry.execute_skill("bad_skill")
    assert "missing execute function" in str(result)

def test_skill_missing_script(mock_skills_dir):
    """Test handling of skill with no script."""
    registry = SkillRegistry(skills_dir=mock_skills_dir)
    result = registry.execute_skill("empty_skill")
    assert result is None

# --- Real Skill Integration Tests ---

def test_real_skills_discovery():
    """Verify that all core skills are loaded in the real registry."""
    real_registry.discover_skills()
    expected = ["calculator", "google_search", "system_info", "todo_list"]
    for skill in expected:
        assert skill in real_registry._skills

def test_calculator_integration():
    """Verify the real calculator skill works."""
    result = real_registry.execute_skill("calculator", expression="2 + 2")
    assert result == "4"
    
    # Test error handling
    result = real_registry.execute_skill("calculator", expression="syntax error")
    assert "Math Error" in result

def test_system_info_integration():
    """Verify the real system_info skill works."""
    result = real_registry.execute_skill("system_info")
    data = json.loads(result)
    assert "os" in data
    assert "cpu_usage_percent" in data
    assert isinstance(data["cpu_usage_percent"], float)
