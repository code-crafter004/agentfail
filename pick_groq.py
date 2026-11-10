import os, requests
from dotenv import load_dotenv
load_dotenv(".env")
r = requests.get("https://api.groq.com/openai/v1/models",
                 headers={"Authorization": "Bearer " + os.getenv("GROQ_API_KEY")})
ids = [m["id"] for m in r.json()["data"]]
print("ALL:", ids)
bad = ("whisper", "guard", "tts", "orpheus", "playai", "safeguard", "compound")
ok = [i for i in ids if not any(b in i.lower() for b in bad)]
def pick(*keys):
    for i in ok:
        if all(k in i.lower() for k in keys):
            return i
choice = pick("llama", "70b") or pick("llama") or pick("gpt-oss") or pick("qwen") or ok[0]
lines = [l for l in open(".env").read().splitlines() if not l.startswith("GROQ_MODEL=")]
lines.append("GROQ_MODEL=" + choice)
open(".env", "w").write("\n".join(lines) + "\n")
print("PICKED:", choice)
