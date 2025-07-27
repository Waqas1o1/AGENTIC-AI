from dotenv import load_dotenv
import os
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

load_dotenv()
set_tracing_disabled(disabled=True)


def get_client(api_key: str, base_url: str) -> AsyncOpenAI:
    if not api_key:
        raise ValueError("API key is missing for the client.")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def build_completion_model(
    client: AsyncOpenAI, model: str = "gemini-2.5-flash"
) -> OpenAIChatCompletionsModel:
    return OpenAIChatCompletionsModel(model=model, openai_client=client)


def create_agent(
    name: str, instructions: str, model: OpenAIChatCompletionsModel
) -> Agent:
    return Agent(name=name, instructions=instructions, model=model)


GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")
GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

OPENAI_API_KEY = os.getenv("CHATGPT_KEY")
OPENAI_API_BASE_URL = "https://api.openai.com/v1"

DEEPSEEK_KEY = "ollama"
DEEPSEEK_KEY_BASE_URL = "http://localhost:11434/v1"
try:
    gimi_client = get_client(GOOGLE_API_KEY, GOOGLE_API_BASE_URL)
    gpt_client = get_client(OPENAI_API_KEY, OPENAI_API_BASE_URL)
    deepseek_client = get_client(DEEPSEEK_KEY, DEEPSEEK_KEY_BASE_URL)
except BaseException as e:
    raise SystemExit(f"❌ Client setup failed: {e}")

llm_model = build_completion_model(gimi_client, model="gemini-2.5-flash")
gpt_model = build_completion_model(gpt_client, model="gpt-4")
deepseek_llm = build_completion_model(deepseek_client, "deepseek-coder")


math_agent = create_agent(
    name="MathAgent",
    instructions="You are a helpful math teacher assistant.",
    model=llm_model,
)
general_agent = create_agent(
    name="GeneralAgent",
    instructions="You are a smart assistant that solves general questions, gives advice, and writes content.",
    model=gpt_model,
)
english_agent = create_agent(
    name="English Agent",
    instructions="You are a helpful AI that correct the english grammer.",
    model=deepseek_llm,
)


if __name__ == "__main__":
    # Run math agent (Gemini)
    # math_query = "Ali has 3 apples. His friend Sara gives him 2 more apples. How many apples does Ali have now?"
    # math_result = Runner.run_sync(math_agent, math_query)
    # print("🧮 MathAgent Response:", math_result.final_output)

    # # Run general agent (GPT)
    # general_query = "Write a short paragraph that discribe agentic ai."
    # general_result = Runner.run_sync(general_agent, general_query)
    # print("📝 GeneralAgent Response:", general_result.final_output)

    # English agent  (Deepseek)
    gramer_query = "Correct grammer of my sentance (She go to the market yesterday to buys some vegetables. The weather were very hot and she forget to take her umbrella. After buying the vegetables, she meet with her friend who was also shopping. They talks for a while before going to their homes. It was a nice day but tiring.)"
    grammer_result = Runner.run_sync(english_agent, gramer_query)
    print("📝 EnglishAgent Response:", grammer_result.final_output)
