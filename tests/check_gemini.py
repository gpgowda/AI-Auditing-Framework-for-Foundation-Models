import sys
from pathlib import Path

# Add the project root directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from models.gemini_wrapper import GeminiWrapper


def test_gemini():

    load_dotenv()

    print("=" * 60)
    print("TESTING GEMINI CONNECTION")
    print("=" * 60)

    try:

        model = GeminiWrapper(
            model_id="gemini-3.6-flash",
            temperature=0.0,
            max_tokens=100,
        )

        print("\nSending test prompt...")

        response = model.query(
            "Say hello and confirm that you are working."
        )

        print("\nResponse:")
        print(response.text)

        print("\nModel:")
        print(response.model_id)

        print("\nSUCCESS: Gemini connection is working.")

    except Exception as error:

        print("\nGEMINI TEST FAILED:")
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    test_gemini()