import uvicorn
from fastapi import FastAPI, WebSocket, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import cv2
import io
from inference_online import OnlineInference
from stream_pipeline_online import StreamSDK
import queue
import threading
import base64
import os
from pathlib import Path
import shutil

app = FastAPI()

# Create output directories
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
temp_dir = output_dir / "temp"
temp_dir.mkdir(exist_ok=True)

# Setup template and static file directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class StreamingServer:
    def __init__(self, sdk_path, source_path):
        self.sdk = StreamSDK(sdk_path, "./checkpoints/ditto_trt_custom")
        self.inference = OnlineInference(
            self.sdk, 
            source_path=source_path,
            output_path=str(temp_dir / "temp_output.mp4"),
            more_kwargs={
                "setup_kwargs": {"online_mode": True},
                "run_kwargs": {"chunksize": (3, 5, 2)}
            }
        )
        
        self.frame_queue = queue.Queue(maxsize=30)
        self.is_running = False
        self.connected_clients = set()
        
        # Override SDK's writer
        self.original_writer = self.sdk.writer
        self.sdk.writer = self.frame_callback
        
    def frame_callback(self, frame, fmt="rgb"):
        if fmt == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        try:
            self.frame_queue.put_nowait(jpg_as_text)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(jpg_as_text)
            except (queue.Empty, queue.Full):
                pass

    async def process_audio(self, audio_chunk):
        if self.is_running:
            self.inference.feed_audio(audio_chunk)
            
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.inference.start()
            
    def stop(self):
        if self.is_running:
            self.is_running = False
            self.inference.stop()
            
    async def get_next_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

# Global server instance
streaming_server = None

# @app.on_event("startup")
# async def startup_event():
#     global streaming_server
#     streaming_server = StreamingServer(
#         sdk_path="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
#         source_path="./data/face.jpg"
#     )

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"source_image_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": str(file_path)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / f"audio_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": str(file_path)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_audio/{audio_filename}")
async def get_audio(audio_filename: str):
    try:
        # Construct the full path to the uploaded audio file
        audio_path = Path("uploads") / audio_filename
        
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_filename}")
        
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename=audio_filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{image_path}/{audio_path}")
async def websocket_endpoint(websocket: WebSocket, image_path: str, audio_path: str = None):
    await websocket.accept()
    
    # Create new streaming server instance for this connection
    server = StreamingServer(
        sdk_path="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
        source_path=f"uploads/{image_path}"
    )
    
    server.connected_clients.add(websocket)
    
    try:
        if len(server.connected_clients) == 1:
            server.start()
            
        while True:
            audio_data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
            
            await server.process_audio(audio_chunk)
            
            frame = await server.get_next_frame()
            if frame is not None:
                # Send both frame and audio data
                await websocket.send_json({
                    "frame": frame,
                    "audio": audio_chunk.tolist()  # Convert numpy array to list for JSON
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        server.connected_clients.remove(websocket)
        if len(server.connected_clients) == 0:
            server.stop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 