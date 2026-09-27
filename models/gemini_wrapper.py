"""
Gemini model wrapper for the AI Auditing Framework.

Uses the Google Gen AI SDK to send prompts to Gemini models
and returns responses in the framework's standard ModelResponse format.
"""

import os

from google import genai
from google.genai import types

from models.base_model import BaseModel, ModelResponse


class GeminiWrapper(BaseModel):
    """
    Wrapper for Google Gemini models.
    """

    def __init__(
        self,
        model_id: str = "gemini-3.6-flash",
        thinking_level: str = "minimal",
        **kwargs,
    ):
        """
        Initialize the Gemini client.

        Parameters
        ----------
        model_id : str
            Gemini model ID.

        thinking_level : str
            Gemini's internal "thinking" tokens count against
            max_output_tokens. With the default budget they used
            almost the whole 300-token limit and cut answers off
            mid-sentence. "minimal" leaves the limit for the
            visible answer, matching GPT and Llama.

        kwargs : dict
            Additional BaseModel parameters such as temperature
            and max_tokens.
        """

        super().__init__(model_id, **kwargs)

        self.thinking_level = thinking_level

        # Safely retrieve the API key.
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. "
                "Please check your .env file."
            )

        # Initialize Gemini client.
        self.client = genai.Client(
            api_key=api_key
        )

    def query(self, prompt: str) -> ModelResponse:
        """
        Send a prompt to Gemini and return a standardized response.

        Parameters
        ----------
        prompt : str
            Prompt to send to Gemini.

        Returns
        -------
        ModelResponse
            Standardized framework response object.
        """

        try:

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.thinking_level,
                    ),
                ),
            )

            # Safely retrieve generated text.
            response_text = response.text

            if not response_text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return ModelResponse(
                text=response_text.strip(),
                model_id=self.model_id,
                prompt=prompt,
                raw=(
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else None
                ),
            )

        except Exception as error:

            raise RuntimeError(
                f"Gemini request failed for model "
                f"'{self.model_id}': {error}"
            ) from error