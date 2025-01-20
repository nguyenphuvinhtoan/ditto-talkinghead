import imageio
import os
import moviepy.editor as mpy
import numpy as np


class VideoWriterByImageIO:
    def __init__(self, video_path, fps=25, **kwargs):
        self.fps = fps
        self.frames = []
        self.audio_chunks = []
        self.samples_per_frame = int(16000 / fps)  # Assuming 16kHz audio
        
    def __call__(self, frame, audio_chunk=None, fmt="bgr"):
        if fmt == "bgr":
            frame = frame[..., ::-1]
        self.frames.append(frame)
        if audio_chunk is not None:
            self.audio_chunks.append(audio_chunk)
            
    def close(self):
        # Combine frames and audio chunks
        # Create video clip from frames
        video_clip = mpy.ImageSequenceClip(self.frames, fps=self.fps)
        
        # Combine audio chunks and create audio clip
        if self.audio_chunks:
            audio_array = np.concatenate(self.audio_chunks)
            audio_clip = mpy.AudioArrayClip(audio_array.reshape(1, -1), fps=16000)
            
            # Combine video and audio
            final_clip = video_clip.set_audio(audio_clip)
        else:
            final_clip = video_clip
            
        # Write to file
        final_clip.write_videofile(self.output_path, 
                                 codec='libx264',
                                 audio_codec='aac',
                                 fps=self.fps)
