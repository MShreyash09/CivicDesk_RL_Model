import os
import sys
import argparse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, current_dir)

from openenv.core.env_server.http_server import create_app
from models import CivicDeskAction, CivicDeskObservation
from civic_desk_environment import CivicDeskEnvironment

# Initialize the FastAPI app
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
    """
    Main entry point for the server. 
    The validator looks for this specific function name.
    """
    # Priority: Environment variable "PORT" -> passed argument -> default 8000
    env_port = os.environ.get("PORT")
    final_port = int(env_port) if env_port else port
    
    uvicorn.run(app, host=host, port=final_port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Civic Desk Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    
    args = parser.parse_args()
    main(host=args.host, port=args.port)
