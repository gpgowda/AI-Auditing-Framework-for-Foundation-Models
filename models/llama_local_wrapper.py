from openai import OpenAI
from models.base_model import BaseModel, ModelResponse


class LlamaLocalWrapper(BaseModel):
    """Talks to a locally-running Ollama server, which exposes an
    OpenAI-compatible endpoint — no API key or internet cost involved.

    Prerequisites (run once, in a terminal):
        1. Install Ollama: https://ollama.com/download
        2. ollama pull llama3.1        (or llama3.1:8b on a lower-spec machine)
        3. Leave Ollama running (it starts a local server on port 11434)
    """

    def __init__(self, model_id: str = "llama3.1", **kwargs):
        super().__init__(model_id, **kwargs)
        self.client = OpenAI(
            api_key="ollama",   # unused, but the client requires a non-empty string
            base_url="http://localhost:11434/v1",
        )

    def query(self, prompt: str) -> ModelResponse:
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        return ModelResponse(
            text=text, model_id=self.model_id, prompt=prompt,
            raw=response.model_dump(),
        )