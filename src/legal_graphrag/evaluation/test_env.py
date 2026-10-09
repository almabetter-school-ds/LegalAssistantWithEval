import os
from dotenv import load_dotenv

# ============================================================
# Load .env
# ============================================================
load_dotenv(override=True)

KEYS = [
    "new_HF_TOKEN",
    "GROQ_API_KEY",
    "GEMINI_API_KEY", 
    "LANGSMITH_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
    "MODEL_NAME",
    "RAGAS_JUDGE_MODEL",
]

print("\n" + "=" * 60)
print("ENVIRONMENT CONFIGURATION TEST")
print("=" * 60)

for key in KEYS:
    value = os.getenv(key)

    if value:
        if "KEY" in key:
            # Don't print the actual secret
            print(f"✅ {key}: SET ({value[:6]}...{value[-4:]})")
        else:
            print(f"✅ {key}: {value}")
    else:
        print(f"❌ {key}: NOT SET")

print("=" * 60)