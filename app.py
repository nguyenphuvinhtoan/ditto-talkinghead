from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
import numpy as np
import io
import cv2
import queue
import threading
from PIL import Image
from stream_pipeline_online import StreamSDK
import librosa
import math

class StreamingSDKWrapper:
    def __init__(self, cfg_pkl, data_root):
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.frame_queue = queue.Queue(maxsize=100)
        self.is_processing = False
        self.current_session = None
        
    async def process_audio_chunk(self, audio_chunk, chunksize=(3, 5, 2)):
        if not self.is_processing:
            raise RuntimeError("Session not started")
            
        # Process audio chunk using SDK
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        if len(audio_chunk) < split_len:
            audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            
        self.sdk.run_chunk(audio_chunk, chunksize)

    def frame_collector_worker(self):
        while self.is_processing:
            try:
                frame = self.sdk.writer_queue.get(timeout=1)
                if frame is None:
                    break
                self.frame_queue.put(frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame collector: {e}")
                break

    async def start_session(self, source_path, setup_kwargs=None):
        if self.is_processing:
            raise RuntimeError("Session already in progress")
            
        if setup_kwargs is None:
            setup_kwargs = {}
            
        # Initialize temporary output path
        output_path = "temp_output.mp4"
        
        # Setup SDK
        self.sdk.setup(source_path, output_path, **setup_kwargs)
        self.sdk.online_mode = True
        
        # Start frame collector thread
        self.is_processing = True
        self.current_session = threading.Thread(target=self.frame_collector_worker)
        self.current_session.start()

    async def end_session(self):
        if not self.is_processing:
            return
            
        self.is_processing = False
        self.sdk.close()
        if self.current_session:
            self.current_session.join()
        
        # Clear queues
        while not self.frame_queue.empty():
            self.frame_queue.get()

app = FastAPI()

sdk_wrapper = StreamingSDKWrapper(
    cfg_pkl="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
    data_root="./checkpoints/ditto_trt_custom"
)

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Receive source image path
        source_path = await websocket.receive_text()
        
        # Start SDK session
        await sdk_wrapper.start_session(source_path)
        
        # Process incoming audio chunks
        while True:
            # Receive audio chunk as bytes
            audio_data = await websocket.receive_bytes()
            if not audio_data:
                break
                
            # Convert to numpy array
            audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
            
            # Process chunk
            await sdk_wrapper.process_audio_chunk(audio_chunk)
            
            # Send generated frames
            while not sdk_wrapper.frame_queue.empty():
                frame = sdk_wrapper.frame_queue.get()
                # Convert frame to bytes
                success, encoded_frame = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if success:
                    await websocket.send_bytes(encoded_frame.tobytes())
                    
    except Exception as e:
        print(f"Error in websocket connection: {e}")
    finally:
        await sdk_wrapper.end_session()