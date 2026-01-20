import os
import re
import importlib.util
import inspect
from typing import Dict, Any, List, Optional
from logger_config import get_logger

logger = get_logger(__name__)

class SkillMetadata:
    """Holds metadata for an AgentSkill."""
    def __init__(self, name: str, description: str, path: str, instructions: str, target_script: Optional[str] = None):
        self.name = name
        self.description = description
        self.path = path
        self.instructions = instructions
        self.target_script = target_script

class SkillRegistry:
    """Registry for discovering and managing AgentSkills."""
    
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = os.path.join(os.getcwd(), skills_dir)
        self._skills: Dict[str, SkillMetadata] = {}
        self.discover_skills()

    def discover_skills(self) -> None:
        """Scans the skills directory for subdirectories with SKILL.md."""
        if not os.path.exists(self.skills_dir):
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for entry in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, entry)
            if os.path.isdir(skill_path):
                skill_md_path = os.path.join(skill_path, "SKILL.md")
                if os.path.exists(skill_md_path):
                    self._load_skill(skill_path, skill_md_path)

    def _load_skill(self, skill_path: str, md_path: str) -> None:
        """Parses SKILL.md and registers the skill."""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Robust Metadata Parser (supports YAML frontmatter and Markdown tables)
            metadata = self._parse_metadata(content)
            
            # Instructions is the body content after the frontmatter/table
            # We'll just take the whole content for the LLM to read
            instructions = content

            name = metadata.get("name") or os.path.basename(skill_path)
            description = metadata.get("description", "No description provided.")

            target_script = self._resolve_script_path(skill_path)

            self._skills[name] = SkillMetadata(name, description, skill_path, instructions, target_script)
            logger.info(f"Discovered skill: {name}")

            # Check for dependencies
            req_path = os.path.join(skill_path, "scripts", "requirements.txt")
            if os.path.exists(req_path):
                logger.info(f"Skill '{name}' has dependencies at {req_path}")

        except Exception as e:
            logger.error(f"Error loading skill at {skill_path}: {e}")

    def _resolve_script_path(self, skill_path: str) -> Optional[str]:
        """Resolves the primary python script for a skill."""
        script_dir = os.path.join(skill_path, "scripts")
        if not os.path.exists(script_dir):
            return None

        # Prefer search.py or main.py or the first .py script
        candidates = ["search.py", "main.py", "run.py"]
        for candidate in candidates:
            p = os.path.join(script_dir, candidate)
            if os.path.exists(p):
                return p

        for file in os.listdir(script_dir):
            if file.endswith(".py"):
                return os.path.join(script_dir, file)

        return None

    def _parse_metadata(self, content: str) -> Dict[str, str]:
        """Parses metadata from YAML frontmatter or Markdown tables."""
        metadata = {}
        # Try YAML Frontmatter
        frontmatter_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if frontmatter_match:
            frontmatter_raw = frontmatter_match.group(1)
            for line in frontmatter_raw.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip().lower()] = val.strip()
            return metadata

        # Fallback: Try Markdown Table (Pattern: | name | value |)
        table_lines = re.findall(r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|", content)
        for key, val in table_lines:
            key = key.strip().lower()
            if key not in ["name", "description", "license"]:
                continue
            metadata[key] = val.strip()
        
        return metadata
    
    def get_skill_instructions(self) -> str:
        """Returns a combined string of all skill instructions for the LLM prompt."""
        if not self._skills:
            return ""
        
        prompt_parts = ["AVAILABLE SKILLS:"]
        # We inject the full SKILL.md content for thorough agent instruction
        for name, meta in self._skills.items():
            prompt_parts.append(f"--- SKILL: {name} ---\n{meta.instructions}\n")
        
        return "\n".join(prompt_parts)

    def execute_skill(self, skill_name: str, **kwargs) -> Optional[str]:
        """Executes a skill's primary script."""
        meta = self._skills.get(skill_name)
        if not meta:
            logger.error(f"Skill '{skill_name}' not found.")
            return None

        if not meta.target_script:
            logger.error(f"No python script found for skill '{skill_name}'")
            return None

        # Strategy 1: Subprocess (Safe, consistent with Anthropic/AgentSkills CLI usage)
        # Strategy 2: Import (Fast, better for local server integration)
        # We'll stick to Import for our own skills, but could fallback to subprocess if import fails.
        return self._run_script(meta.target_script, **kwargs)

    def _run_script(self, script_path: str, **kwargs) -> Optional[str]:
        """Runs a python script dynamically."""
        try:
            spec = importlib.util.spec_from_file_location("skill_module", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Convention: each script should have an execute or execute_search function
            # To be flexible, we can look for 'execute_search' or generic 'execute'
            func_name = None
            for name in ["execute", "execute_search", "run"]:
                if hasattr(module, name):
                    func_name = name
                    break
            
            if not func_name:
                error_msg = f"Script {script_path} missing execute function."
                logger.error(error_msg)
                return error_msg
            
            func = getattr(module, func_name)
            
            # Robust Argument Filtering: Only pass kwargs that the function accepts
            sig = inspect.signature(func)
            filtered_kwargs = {
                k: v for k, v in kwargs.items() 
                if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            }
            
            return func(**filtered_kwargs)

        except Exception as e:
            logger.error(f"Error executing script {script_path}: {e}")
            return f"Error: {e}"

# Singleton instance
registry = SkillRegistry()
