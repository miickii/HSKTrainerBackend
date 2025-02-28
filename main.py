import os
import io
import json
import time
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import random

import numpy as np
import wave
import uvicorn
import torch
import opencc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
from hsk import HSKManager  # Assumes your hsk.py defines HSKManager

import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Modify database path for Vercel
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', 'hsk.db')

# Initialize FastAPI app
app = FastAPI(
    title="HSK Master API",
    description="Backend API for HSK Master Chinese learning application",
    version="1.0.0"
)

# =============== Models ===============

class SampleSentence(BaseModel):
    simplified: str
    pinyin: str
    english: str

class SentenceRequest(BaseModel):
    hsk_levels: List[int] = [1, 2, 3]
    count: int = 5
    exclude_ids: List[int] = []

class WordUpdateResult(BaseModel):
    id: Optional[int]
    word: str
    correct: bool

class PracticeResult(BaseModel):
    word_id: int
    was_correct: bool

class VocabularyFilter(BaseModel):
    levels: Optional[List[int]] = None
    filter_type: Optional[str] = None  # "all", "mastered", "learning", "favorite"
    search: Optional[str] = None

# =============== Setup ===============

# Setup Chinese traditional to simplified converter
converter = opencc.OpenCC('t2s')

# Initialize Whisper model for transcription
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="chinese", task="transcribe")
forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="chinese", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

# Determine device - use MPS for Mac with Apple Silicon, CUDA for Nvidia GPU, or CPU
device = "cpu"

print(f"Using device: {device}")

# Load the model on the appropriate device
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)

# Initialize HSK Manager
manager = HSKManager()

# Configure CORS for frontend
origins = [
    "https://miickii.github.io/HSKTrainer/",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://192.168.1.*",  # Allow local network access for mobile testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============== OpenAI API Setup ===============

# Load API keys from environment variables or a configuration file
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Import OpenAI library and initialize clients
try:
    from openai import OpenAI

    # Initialize clients
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        openai_client = None

except ImportError:
    print("OpenAI library not installed. Some features will be disabled.")
    openai_client = None

# =============== Helper Functions ===============

def get_completion_cost(completion):
    try:
        return completion.usage.prompt_tokens * 0.15 / 1000000 + completion.usage.completion_tokens * 0.6 / 1000000
    except:
        return "unknown"

def get_example_sentence(manager, word):
    word_id = word.get("id")
    examples = []
    if word_id:
        word_obj = manager.get_word_by_id(word_id)
        if word_obj and word_obj.examples:
            try:
                # Parse the examples JSON
                examples = json.loads(word_obj.examples)
            except Exception as e:
                print(f"Error parsing examples for word {word_id}: {str(e)}")

    # If we have words with examples, select a random one
    if examples:
        print(examples)
        # Select a random word with examples
        selected_example = random.choice(examples)
        
        # Format the sentence
        sentence = {
            "simplified": selected_example.get("simplified", ""),
            "pinyin": selected_example.get("pinyin", ""),
            "english": selected_example.get("english", ""),
            "source_word": word.get("simplified")
        }
        print(f"Selected example sentence: {sentence.get('simplified', 'N/A')}")
        return sentence
    
    print("No examples found")
    return {"simplified": "", "pinyin": "", "english": "", "source_word": word.simplified}

def save_audio_to_bytes(samples):
    """Convert float audio samples to WAV format bytes."""
    data = np.array(samples, dtype=np.float32)
    audio_int16 = (data * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(2)      # 2 bytes per sample for int16
        wf.setframerate(16000)
        wf.writeframes(audio_int16.tobytes())
    byte_io.seek(0)
    return byte_io

def transcribe_audio_local(audio_samples):
    """Transcribe audio using local Whisper model."""
    try:
        audio_array = np.array(audio_samples, dtype=np.float32)
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")

        # Move input tensors to the same device as the model
        input_features = inputs.input_features.to(device)
        if "attention_mask" in inputs:
            attention_mask = inputs.attention_mask.to(device)
        else:
            # Create a mask of ones
            attention_mask = torch.ones(
                input_features.shape[:2], dtype=torch.long, device=device
            )

        # Generate transcription
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids
        )

        # Decode the token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Convert to simplified Chinese if needed
        simplified_text = converter.convert(transcription)
        return simplified_text
        
    except Exception as e:
        print(f"Error in local transcription: {e}")
        return None

def transcribe_audio_openai(audio_file):
    """Transcribe audio using OpenAI's Whisper API."""
    try:
        if not openai_client:
            return None
            
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="zh"
        )
        simplified_text = converter.convert(transcription.text)
        return simplified_text
        
    except Exception as e:
        print(f"Error in OpenAI transcription: {e}")
        return None

def get_transcription(audio_samples, use_local=True):
    """Get transcription from audio samples, with fallback options."""
    audio_file = save_audio_to_bytes(audio_samples)
    audio_file.name = "audio.wav"
    
    # First try local transcription if enabled
    if use_local:
        result = transcribe_audio_local(audio_samples)
        if result:
            return result
    
    # If local fails or is disabled, try OpenAI if available
    if openai_client:
        result = transcribe_audio_openai(audio_file)
        if result:
            return result
    
    # If all transcription methods fail
    if use_local:
        # Return error message
        raise Exception("Transcription failed with all available methods")
    else:
        # Fall back to local as last resort
        return transcribe_audio_local(audio_samples)

def process_transcription(transcription, sampled_words):
    result = manager.update_words(transcription, sampled_words)
    
    # Log transcription quality information
    print(f"Transcription: '{transcription}'")
    
    # Print some debug info
    matched_words = [r["word"] for r in result if r["correct"]]
    if matched_words:
        print(f"Words matched in transcription: {', '.join(matched_words)}")
    
    # Return formatted results
    return result

def save_debug_audio(samples, filename):
    """Save debug audio file."""
    try:
        data = np.array(samples, dtype=np.float32)
        data_int16 = (data * 32767).astype(np.int16)
        os.makedirs("debug_audio", exist_ok=True)
        with wave.open(f"debug_audio/{filename}", 'w') as wf:
            wf.setnchannels(1)      # Mono
            wf.setsampwidth(2)      # 16-bit samples
            wf.setframerate(16000)  # 16 kHz sample rate
            wf.writeframes(data_int16.tobytes())
    except Exception as e:
        print(f"Error saving debug audio: {e}")

# =============== WebSocket Connection Manager ===============

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"New WebSocket connection: {client_id}")
        print("connection open")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"WebSocket disconnected: {client_id}")
            print("connection closed")
            
    def get_connection(self, client_id: str) -> Optional[WebSocket]:
        return self.active_connections.get(client_id)

# Initialize connection manager
connection_manager = ConnectionManager()

# =============== API Routes ===============

@app.get("/")
async def root():
    return {"message": "Welcome to HSK Master API"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/vocabulary")
async def get_vocabulary(
    level: Optional[int] = None,
    filter_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 1000
):
    """Get vocabulary words with optional filtering."""
    try:
        # Get words from HSK manager
        words = manager.get_all_words()
        
        # Apply level filter
        if level is not None:
            words = [w for w in words if w.level == level]
            
        # Apply filter type
        if filter_type == "mastered":
            words = [w for w in words if w.correct_count > 0]
        elif filter_type == "learning":
            words = [w for w in words if w.correct_count == 0]
        elif filter_type == "favorite":
            words = [w for w in words if getattr(w, 'is_favorite', False)]
            
        # Apply search filter
        if search:
            search_lower = search.lower()
            words = [w for w in words if 
                     search in w.simplified or 
                     search_lower in w.pinyin.lower() or 
                     search_lower in w.meanings.lower()]
            
        # Limit results
        words = words[:limit]
        
        # Convert to dictionary for JSON response
        result = []
        for word in words:
            word_dict = word.__dict__.copy()
            # Remove SQLAlchemy internal attributes
            word_dict.pop('_sa_instance_state', None)
            result.append(word_dict)
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving vocabulary: {str(e)}")

@app.get("/api/download-vocabulary")
async def download_vocabulary(level: Optional[int] = None):
    """Generate a JSON file of vocabulary words that can be used offline."""
    try:
        # Debug print
        print(f"Fetching vocabulary, level filter: {level}")
        
        # Get all words directly from the database
        words = manager.get_all_words()
        
        # Apply level filter if specified
        if level is not None:
            words = [w for w in words if w.level == level]
        
        # Convert to dictionary for JSON response
        result = []
        for word in words:
            try:
                # Get word attributes safely
                word_dict = {
                    "id": getattr(word, 'id', None),
                    "simplified": getattr(word, 'simplified', ""),
                    "traditional": getattr(word, 'traditional', ""),
                    "pinyin": getattr(word, 'pinyin', ""),
                    "meanings": getattr(word, 'meanings', ""),
                    "level": getattr(word, 'level', 1),
                    "radical": getattr(word, 'radical', ""),
                    "pos": getattr(word, 'pos', ""),
                    "frequency": getattr(word, 'frequency', 0),
                    
                    # Use camelCase for frontend compatibility
                    "correctCount": getattr(word, 'correct_count', 0),
                    "incorrectCount": getattr(word, 'incorrect_count', 0),
                    
                    # Keep snake_case versions for backward compatibility
                    "correct_count": getattr(word, 'correct_count', 0),
                    "incorrect_count": getattr(word, 'incorrect_count', 0),
                    
                    # SRS fields
                    "srsLevel": getattr(word, 'srs_level', 0),
                    "nextReview": getattr(word, 'next_review', datetime.now().strftime('%Y-%m-%d')),
                    "lastReviewed": getattr(word, 'last_reviewed', None),
                    "isFavorite": getattr(word, 'is_favorite', False)
                }
                result.append(word_dict)
            except Exception as e:
                print(f"Error processing word: {e}")
                # Skip problematic words
                continue
        
        # Debug print
        print(f"Returning {len(result)} vocabulary words")
        return result
        
    except Exception as e:
        print(f"Error in download_vocabulary: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error downloading vocabulary: {str(e)}")
    
@app.post("/api/reset-all-words")
async def reset_all_words(level: Optional[int] = None):
    """Reset all words to initial state (learning status)"""
    try:
        # Use the HSK manager's reset method
        reset_count = manager.reset_all_words(level)
        
        return {
            "message": f"Successfully reset {reset_count} words",
            "count": reset_count
        }
        
    except Exception as e:
        print(f"Error resetting words: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error resetting words: {str(e)}")

@app.post("/api/update-word/{word_id}")
async def update_word(word_id: int, was_correct: bool):
    """Update a word's SRS information."""
    try:
        # Get the word
        word = manager.get_word_by_id(word_id)
        if not word:
            raise HTTPException(status_code=404, detail=f"Word with ID {word_id} not found")
            
        # Update SRS level
        current_level = getattr(word, 'srs_level', 0)
        
        # SRS intervals in days
        intervals = [1, 3, 7, 14, 30, 60, 120, 240]
        
        if was_correct:
            # Move to next level (max at highest interval)
            next_level = min(current_level + 1, len(intervals) - 1)
        else:
            # Reset or step back
            next_level = max(0, current_level - 2)
            
        days_until_next_review = intervals[next_level]
        next_review_date = datetime.now() + timedelta(days=days_until_next_review)
        
        # Update the word
        word.srs_level = next_level
        word.next_review = next_review_date.date()
        
        if was_correct:
            word.correct_count = (word.correct_count or 0) + 1
        else:
            # Add incorrect_count field if it doesn't exist
            if hasattr(word, 'incorrect_count'):
                word.incorrect_count = (word.incorrect_count or 0) + 1
                
        # Commit changes
        manager.session.commit()
        
        return {
            "id": word.id,
            "srs_level": word.srs_level,
            "next_review": word.next_review.isoformat() if hasattr(word.next_review, 'isoformat') else word.next_review,
            "correct_count": word.correct_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating word: {str(e)}")

@app.post("/api/toggle-favorite/{word_id}")
async def toggle_favorite(word_id: int):
    """Toggle a word's favorite status."""
    try:
        # Get the word
        word = manager.get_word_by_id(word_id)
        if not word:
            raise HTTPException(status_code=404, detail=f"Word with ID {word_id} not found")
            
        # Toggle favorite status
        current = getattr(word, 'is_favorite', False)
        word.is_favorite = not current
        
        # Commit changes
        manager.session.commit()
        
        return {
            "id": word.id,
            "is_favorite": word.is_favorite
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error toggling favorite: {str(e)}")

@app.post("/api/transcribe")
async def transcribe(request: Request):
    """Transcribe audio via REST API (alternative to WebSocket)."""
    try:
        # Get file content
        content = await request.body()
        
        # Convert to in-memory file
        audio_file = io.BytesIO(content)
        audio_file.name = "audio.wav"
        
        # Transcribe with OpenAI
        if openai_client:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="zh"
            )
            simplified_text = converter.convert(transcription.text)
            
            return {"transcription": simplified_text}
        else:
            raise HTTPException(status_code=501, detail="OpenAI client not configured")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# =============== WebSocket Routes ===============

@app.websocket("/ws/api")
async def websocket_endpoint(websocket: WebSocket):
    # Generate client ID
    client_id = str(uuid.uuid4())
    
    # Accept connection
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Wait for message with timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60  # 60 second timeout
                )
                print(f"Received WebSocket message: {data.get('type', 'unknown')}")
                
                msg_type = data.get("type")
                
                if msg_type == "get_sample_words":
                    # Get parameters
                    hsk_levels = data.get("hsk_levels", [1, 2, 3])
                    exclude_ids = data.get("exclude_ids", [])
                    count = 10
                    
                    print(f"Sampling words: levels={hsk_levels}, count={count}")
                    
                    # Sample words from database
                    possible_words = []
                    
                    # First try to get words due for review
                    try:
                        # This assumes you've added next_review field to your ChineseWord model
                        due_words = manager.get_words_due_for_review(count=count, levels=hsk_levels)
                        if due_words:
                            possible_words = due_words
                            print(f"Found {len(due_words)} words due for review")
                    except Exception as e:
                        print(f"Error getting due words: {str(e)}")
                        # Fallback to regular sampling if review functionality not implemented
                        pass
                    
                    # If we don't have enough words from due reviews, use regular sampling
                    if len(possible_words) < count:
                        try:
                            additional_words = manager.sample_words(
                                count - len(possible_words),
                                min(hsk_levels) if hsk_levels else 1
                            )
                            print(f"Sampled {len(additional_words)} additional words")
                            possible_words.extend(additional_words)
                        except Exception as e:
                            print(f"Error sampling words: {str(e)}")
                            # Return an error if we can't get any words
                            if not possible_words:
                                await websocket.send_json({
                                    "type": "error",
                                    "detail": f"Failed to sample words: {str(e)}"
                                })
                                continue
                    
                    selected_word = random.choice(possible_words)
                    print()
                    print(selected_word)
                    # Get a sentence from examples instead of generating with LLM
                    sentence = get_example_sentence(manager, selected_word)
                    print(sentence)
                    # Send response
                    await websocket.send_json({
                        "type": "sample_sentence",
                        "sentence": sentence,
                        "sampled_words": [selected_word]
                    })
                    print("Sent sample sentence response")
                    
                elif msg_type == "upload_audio":
                    start_time = time.time()
                    
                    # Extract audio data
                    realtime_input = data.get("realtime_input", {})
                    media_chunks = realtime_input.get("media_chunks", [])
                    
                    if not media_chunks:
                        await websocket.send_json({
                            "type": "error",
                            "detail": "No audio data provided"
                        })
                        continue
                    
                    # Collect all audio samples
                    all_samples = []
                    for chunk in media_chunks:
                        all_samples.extend(chunk.get("samples", []))
                    
                    if not all_samples:
                        await websocket.send_json({
                            "type": "error",
                            "detail": "Empty audio data"
                        })
                        continue
                    
                    # For debugging
                    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # save_debug_audio(all_samples, f"ws_audio_{timestamp}.wav")
                    
                    # Transcribe audio
                    try:
                        transcription_text = get_transcription(all_samples)
                        
                        # Get sampled words from the client (if provided)
                        sampled_words = data.get("sampled_words", [])
                        
                        # Process transcription
                        update_results = process_transcription(transcription_text, sampled_words)
                        
                        # Calculate processing time
                        processing_time = time.time() - start_time
                        
                        # Send response
                        await websocket.send_json({
                            "type": "audio_upload_ack",
                            "transcription": transcription_text,
                            "update_results": update_results,
                            "processing_time_ms": round(processing_time * 1000)
                        })
                    except Exception as e:
                        print(f"Transcription error: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "detail": f"Transcription error: {str(e)}"
                        })
                elif msg_type == "heartbeat":
                    # Respond with a pong message to keep the connection alive
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "detail": f"Unknown message type: {msg_type}"
                    })
                    
            except asyncio.TimeoutError:
                # Send a ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    # Connection probably closed
                    break
                
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        import traceback
        traceback.print_exc()
        connection_manager.disconnect(client_id)

# =============== Static Files ===============

# Serve static files for the PWA
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """
    Redirect all other requests to the API routes or return a message.
    This handles the case when the frontend is served separately.
    """
    return {
        "message": "HSK Master API is running. Frontend should be served separately during development.",
        "documentation": "/docs",
        "status": "ok"
    }
