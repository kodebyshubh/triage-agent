import os

# Runs before any test module imports app code. load_dotenv() never overrides
# variables that are already set, so these keep unit tests from sending traces to
# LangSmith even when a real key sits in .env.
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"
