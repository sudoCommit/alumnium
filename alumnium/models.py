from enum import Enum


class Model(Enum):
    ANTHROPIC = "claude-3-5-sonnet-20240620"
    GOOGLE = "gemini-1.5-flash"
    OPENAI = "gpt-4o-mini"
