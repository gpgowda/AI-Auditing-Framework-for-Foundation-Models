import os
from openai import OpenAI
from models.base_model import BaseModel, ModelResponse


class GPTWrapper(BaseModel):
    def __init__(self, model_id: str = "gpt-4o", **kwargs):
        super().__init__(model_id, **kwargs)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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