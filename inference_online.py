import librosa
import math
import os
import numpy as np
import random
import torch
import pickle
import queue
import threading
import time
from stream_pipeline_online import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


class OnlineInference:
    def __init__(self, sdk: StreamSDK, source_path: str, output_path: str, more_kwargs: str | dict = {}):
        if isinstance(more_kwargs, str):
            more_kwargs = load_pkl(more_kwargs)
        self.setup_kwargs = more_kwargs.get("setup_kwargs", {})
        self.run_kwargs = more_kwargs.get("run_kwargs", {})
        
        # Ensure online mode is set
        self.setup_kwargs["online_mode"] = True
        
        # Initialize SDK and setup
        self.sdk = sdk
        self.sdk.setup(source_path, output_path, **self.setup_kwargs)
        
        # Audio processing parameters
        self.chunksize = self.run_kwargs.get("chunksize", (3, 5, 2))
        self.split_len = int(sum(self.chunksize) * 0.04 * 16000) + 80
        
        # Setup audio buffer
        self.audio_buffer = np.zeros((self.chunksize[0] * 640,), dtype=np.float32)
        
        # Processing control
        self.is_running = False
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Setup processing thread
        self.process_thread = threading.Thread(target=self._process_audio_chunks)
        
    def start(self):
        """Start the online processing"""
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
            self.process_thread.start()
            
    def stop(self):
        """Stop the online processing"""
        if self.is_running:
            self.stop_event.set()
            self.audio_queue.put(None)  # Signal to stop processing
            self.process_thread.join()
            self.is_running = False
            self.sdk.close()
            
    def feed_audio(self, audio_chunk):
        """Feed new audio chunk for processing"""
        if self.is_running:
            self.audio_queue.put(audio_chunk)
            
    def _process_audio_chunks(self):
        while not self.stop_event.is_set():
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            if audio_chunk is None:
                break
                
            # Concatenate with buffer
            audio = np.concatenate([self.audio_buffer, audio_chunk])
            
            # Process chunks
            for i in range(0, len(audio), self.chunksize[1] * 640):
                chunk = audio[i:i + self.split_len]
                if len(chunk) < self.split_len:
                    chunk = np.pad(chunk, (0, self.split_len - len(chunk)), mode="constant")
                self.sdk.run_chunk(chunk, self.chunksize)
                
            # Update buffer with remaining audio
            buffer_size = self.chunksize[0] * 640
            if len(audio) > buffer_size:
                self.audio_buffer = audio[-buffer_size:]
            else:
                self.audio_buffer = audio


def run_online_stream(sdk_path, source_path, output_path, audio_stream, more_kwargs={}):
    """
    Main function to run online inference with streaming audio
    
    Parameters:
        sdk_path: Path to SDK configuration pickle file
        source_path: Path to source image/video
        output_path: Path for output video
        audio_stream: Generator/Iterator providing audio chunks
        more_kwargs: Additional configuration parameters
    """
    sdk = StreamSDK(sdk_path, "./checkpoints/ditto_trt_custom")
    inference = OnlineInference(sdk, source_path, output_path, more_kwargs)
    
    # Create temporary audio file
    temp_audio_path = output_path + ".tmp.wav"
    audio_chunks = []
    
    try:
        inference.start()
        for audio_chunk in audio_stream:
            inference.feed_audio(audio_chunk)
            audio_chunks.append(audio_chunk)
            
    finally:
        inference.stop()
        
        # Save collected audio chunks to temporary file
        if audio_chunks:
            import soundfile as sf
            audio_data = np.concatenate(audio_chunks)
            sf.write(temp_audio_path, audio_data, 16000)
        
        # Combine with audio using ffmpeg
        if os.path.exists(inference.sdk.tmp_output_path):
            cmd = f'ffmpeg -loglevel error -y -i "{inference.sdk.tmp_output_path}" -i "{temp_audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
            os.system(cmd)
            
            # Cleanup temporary files
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
