# src/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your frontend to fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# LiveKit credentials
LIVEKIT_SERVER_URL = "ws://127.0.0.1:7880"
LIVEKIT_API_KEY = "devkey"
LIVEKIT_API_SECRET = "secret"

APP_CONFIG_DEFAULTS = {
    "accent": "#6366f1",
    "accentDark": "#4f46e5",
    # add any other default app config
}

@app.get("/config/{sandbox_id}")
async def get_config(sandbox_id: str):
    return {
        "sandboxId": sandbox_id,
        "livekitServerUrl": LIVEKIT_SERVER_URL,
        "livekitApiKey": LIVEKIT_API_KEY,
        "livekitApiSecret": LIVEKIT_API_SECRET,
        **APP_CONFIG_DEFAULTS,
    }
