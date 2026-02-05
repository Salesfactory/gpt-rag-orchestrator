import os

# IF YOU EVER UPDATE ANYTHING IN CLAUDE_SKILLS, you should run maintain_versioning to update skill
from anthropic import Anthropic
from anthropic.lib import files_from_dir
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BETAS = ["skills-2025-10-02"]


class SkillClient:
    def __init__(
        self, client: Anthropic | None = None, betas: list[str] | None = None
    ) -> None:
        self._client = client or Anthropic()
        self._betas = betas or DEFAULT_BETAS

    def create_skill(self, display_title: str, files_dir: str):
        return self._client.beta.skills.create(
            display_title=display_title,
            files=files_from_dir(files_dir),
            betas=self._betas,
        )

    def maintain_versioning(self, skill_id: str, files_dir: str):
        """Update the deployed skill after local changes."""
        return self._client.beta.skills.versions.create(
            skill_id=skill_id,
            files=files_from_dir(files_dir),
            betas=self._betas,
        )


if __name__ == "__main__":
    client = SkillClient()

    # skill_creation = client.create_skill(
    #     display_title="Creative Brief",
    #     files_dir="creative-brief")
    # print(skill_creation)

    version = client.maintain_versioning(
        skill_id=os.getenv("ANTHROPIC_SKILL_ID"),
        files_dir="CLAUDE_SKILLS/creative-brief",
    )
    print(version.id)
