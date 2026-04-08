import os
import sys
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, current_dir)

from openenv.core.env_server.http_server import create_app
from models import CivicDeskAction, CivicDeskObservation
from civic_desk_environment import CivicDeskEnvironment

app = create_app(
    CivicDeskEnvironment,
    CivicDeskAction,
    CivicDeskObservation,
    env_name="civic_desk",
    max_concurrent_envs=1,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "online"}

def main(host: str = "0.0.0.0", port: int = 8000):
    # This reads the port from Hugging Face automatically
    final_port = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=final_port)

if __name__ == '__main__':
    main()
