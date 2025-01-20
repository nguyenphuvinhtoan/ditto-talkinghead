# server.py
from http.client import HTTPException
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import tempfile
import os
import librosa
from stream_pipeline_online import StreamSDK
import asyncio
import queue
from typing import Iterator
import io
import json
import base64

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class StreamingSDKWrapper:
    def __init__(self, cfg_pkl, data_root):
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.frame_queue = queue.Queue(maxsize=100)
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_processing = False
        
    async def process_audio(self, audio: np.ndarray, source_path: str, setup_kwargs=None):
        if setup_kwargs is None:
            setup_kwargs = {}
            
        # Create temporary output path
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_output:
            output_path = tmp_output.name
            
        # Setup SDK
        self.sdk.setup(source_path, output_path, **setup_kwargs)
        self.sdk.online_mode = True
        
        # Process audio in chunks
        chunksize = (3, 5, 2)  # Default chunksize
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            
            self.sdk.run_chunk(audio_chunk, chunksize)
            
            # Collect frames from writer_queue
            while not self.sdk.writer_queue.empty():
                frame = self.sdk.writer_queue.get()
                if frame is not None:
                    self.frame_queue.put(frame)
                    
        self.sdk.close()
        
        # Cleanup temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)

    def get_frame(self) -> np.ndarray | None:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def frame_collector_worker(self):
        while self.is_processing:
            try:
                item = self.sdk.writer_queue.get(timeout=1)
                if item is None:
                    break
                    
                if isinstance(item, tuple):
                    frame, audio_chunk = item
                    self.frame_queue.put(frame)
                    self.audio_queue.put(audio_chunk)
                else:
                    self.frame_queue.put(item)
            except queue.Empty:
                continue

    async def get_frame_and_audio(self):
        try:
            frame = self.frame_queue.get_nowait()
            audio = self.audio_queue.get_nowait()
            return frame, audio
        except queue.Empty:
            return None, None

sdk_wrapper = StreamingSDKWrapper(
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
    data_root="./checkpoints/ditto_trt_custom"
)

def frame_generator(sdk_wrapper: StreamingSDKWrapper) -> Iterator[bytes]:
    frame_count = 0
    while True:
        frame = sdk_wrapper.get_frame()
        if frame is None:
            print(f"Total frames processed: {frame_count}")
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        success, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if success:
            print(f"Frame {frame_count} encoded successfully")
            frame_data = encoded_frame.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n'
                b'\r\n' + frame_data + b'\r\n'
            )
        else:
            print(f"Failed to encode frame {frame_count}")

@app.post("/process")
async def process_audio_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    try:
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            content = await image.read()
            tmp_img.write(content)
            source_path = tmp_img.name
        
        # Read and process audio file
        audio_content = await audio.read()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio.write(audio_content)
            audio_path = tmp_audio.name
        
        # Load audio using librosa
        audio_data, sr = librosa.load(audio_path, sr=16000)
        
        # Process audio through SDK
        await sdk_wrapper.process_audio(audio_data, source_path)
        
        # Cleanup temporary files
        os.unlink(source_path)
        os.unlink(audio_path)
        
        # Return streaming response
        headers = {
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Pragma': 'no-cache'
        }
        
        return StreamingResponse(
            frame_generator(sdk_wrapper),
            media_type='multipart/x-mixed-replace; boundary=frame',
            headers=headers
        )
        
    except Exception as e:
        print(f"Error in process_audio_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream")
async def stream_response():
    async def event_generator():
        while True:
            frame, audio = await sdk_wrapper.get_frame_and_audio()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
                
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Convert audio to base64
            audio_base64 = base64.b64encode(audio).decode('utf-8')
            
            # Create event data
            data = {
                'frame': frame_base64,
                'audio': audio_base64
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            # Control frame rate
            await asyncio.sleep(1/25)  # 25 FPS

    return Response(
        event_generator(),
        media_type="text/event-stream"
    )

@app.get("/")
async def read_root():
    return {"message": "Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)