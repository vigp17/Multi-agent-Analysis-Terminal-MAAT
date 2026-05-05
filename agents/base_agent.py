from __future__ import annotations

import json
import anthropic
from abc import ABC, abstractmethod
from config.settings import settings


class BaseAgent(ABC):

    def __init__(self, name: str):
        self.name = name
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def analyze(self, ticker: str, data: dict) -> dict: ...

    def _call_claude(self, user_prompt: str) -> str:
        msg = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=self.system_prompt(),
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text

    def _call_claude_json(self, user_prompt: str) -> dict:
        raw = self._call_claude(user_prompt)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"[{self.name}] Bad JSON response: {e}\n{raw[:400]}")
        
    def rebut(self, opposing_report: dict, conflict_description: str) -> str:
        prompt = f"""
        The CIO flagged a conflict: {conflict_description}
        You are the {self.name}. Your signal is based on your area of expertise.
        Defend your position in 3-5 sentences using evidence from your domain only.
        This is a live trading desk debate.
        """
        msg = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=500,
            temperature=settings.temperature,
            system="You are a senior analyst on a trading desk defending your investment thesis. Respond in plain English prose only. Do not use JSON, markdown, or bullet points.",
            messages=[{"role": "user", "content": prompt}],
    )
        return msg.content[0].text

    def _format_data_block(self, data: dict) -> str:
        return json.dumps(data, indent=2, default=str)

    def __repr__(self):
        return f"<{self.__class__.__name__} model={settings.claude_model}>"