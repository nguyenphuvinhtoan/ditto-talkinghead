let ws = null;
let mediaStream = null;
let audioContext = new (window.AudioContext || window.webkitAudioContext)();
let scriptProcessor = null;
let sourceImagePath = null;
let sourceAudioPath = null;
let audioPlaybackContext = null;
let audioPlaybackSource = null;
let audioPlaybackProcessor = null;
let audioQueue = [];
let isPlayingAudio = false;

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const statusText = document.getElementById('statusText');
const canvas = document.getElementById('videoCanvas');
const ctx = canvas.getContext('2d');
const imageUpload = document.getElementById('imageUpload');
const audioUpload = document.getElementById('audioUpload');
const imageStatus = document.getElementById('imageStatus');
const audioStatus = document.getElementById('audioStatus');

imageUpload.addEventListener('change', uploadImage);
audioUpload.addEventListener('change', uploadAudio);
startButton.addEventListener('click', startStreaming);
stopButton.addEventListener('click', stopStreaming);

async function uploadImage(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        imageStatus.textContent = 'Uploading...';
        const response = await fetch('/upload/image', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        sourceImagePath = data.filename.split('/').pop();
        imageStatus.textContent = 'Upload successful';
        startButton.disabled = false;
        statusText.textContent = 'Status: Ready to stream';
    } catch (error) {
        imageStatus.textContent = 'Upload failed: ' + error.message;
    }
}

async function uploadAudio(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        audioStatus.textContent = 'Uploading...';
        const response = await fetch('/upload/audio', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        sourceAudioPath = data.filename.split('/').pop();
        audioStatus.textContent = 'Upload successful';
    } catch (error) {
        audioStatus.textContent = 'Upload failed: ' + error.message;
    }
}

async function startStreaming() {
    if (!sourceImagePath) {
        statusText.textContent = 'Status: Please upload source image first';
        return;
    }
    
    try {
        // Create new audio contexts
        audioContext = new AudioContext();
        audioPlaybackContext = new AudioContext();
        
        // If we have an uploaded audio file, use it
        if (sourceAudioPath) {
            const response = await fetch(`/get_audio/${sourceAudioPath}`);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Create audio source and processor
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);
            
            // Connect audio nodes
            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);
            
            // Setup WebSocket with a new connection
            const wsPath = `ws://${window.location.host}/ws/${sourceImagePath}/${sourceAudioPath}`;
            ws = new WebSocket(wsPath);
            
            ws.onopen = () => {
                statusText.textContent = 'Status: Connected';
                startButton.disabled = true;
                stopButton.disabled = false;
                // Start playing audio when connection is established
                source.start(0);
            };
            
            ws.onclose = () => {
                stopStreaming();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                // Handle video frame
                const img = new Image();
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                };
                img.src = 'data:image/jpeg;base64,' + data.frame;
                
                // Handle audio data
                if (data.audio) {
                    const audioData = new Float32Array(data.audio);
                    audioQueue.push(audioData);
                    playNextAudioChunk();
                }
            };
            
            // Process audio chunks
            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                    ws.send(inputData.buffer);
                }
            };
        } else {
            statusText.textContent = 'Status: Please upload an audio file';
            return;
        }
        
    } catch (error) {
        console.error('Error:', error);
        statusText.textContent = 'Status: Error - ' + error.message;
    }
}

function playNextAudioChunk() {
    if (isPlayingAudio || audioQueue.length === 0) return;
    
    isPlayingAudio = true;
    const audioData = audioQueue.shift();
    
    const audioBuffer = audioContext.createBuffer(1, audioData.length, 16000);
    audioBuffer.getChannelData(0).set(audioData);
    
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    
    source.onended = () => {
        isPlayingAudio = false;
        playNextAudioChunk();
    };
    
    source.start();
}

function stopStreaming() {
    // Close WebSocket connection
    if (ws) {
        ws.close();
        ws = null;
    }
    
    // Stop media stream if exists
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    // Disconnect and cleanup script processor
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    
    // Close and cleanup audio contexts
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    
    if (audioPlaybackContext) {
        audioPlaybackContext.close();
        audioPlaybackContext = null;
    }
    
    // Clear audio queue
    audioQueue = [];
    isPlayingAudio = false;
    
    // Reset UI
    startButton.disabled = false;
    stopButton.disabled = true;
    statusText.textContent = 'Status: Ready';
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
} 