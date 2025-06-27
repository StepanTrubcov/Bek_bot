import os
import uuid
import gc
import logging
import signal
import sys
import psutil
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ==================== Configuration ====================
class Config:
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    AUDIO_SAMPLE_RATE = 48000
    MAX_AUDIO_DURATION = 120  # seconds
    PROCESSING_TIMEOUT = 30  # seconds
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    RATE_LIMIT = "5 per minute"

# ==================== Initialization ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=[Config.RATE_LIMIT]
)

# ==================== Audio Storage ====================
class AudioStorage:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
        self.references = {}
    
    def get_reference_path(self, teacher_id):
        path = self.references.get(str(teacher_id))
        return path if path and os.path.exists(path) else None
    
    def save_reference(self, teacher_id, file):
        try:
            # Remove old reference if exists
            if teacher_id in self.references:
                self._safe_remove(self.references[teacher_id])
            
            filename = f"ref_{teacher_id}_{uuid.uuid4()}.wav"
            filepath = os.path.join(self.upload_folder, filename)
            
            audio = self._load_and_validate_audio(file)
            sf.write(filepath, audio, Config.AUDIO_SAMPLE_RATE)
            
            self.references[teacher_id] = filepath
            return filepath
        except Exception as e:
            logger.error(f"Failed to save reference: {str(e)}")
            raise

    def _load_and_validate_audio(self, file):
        temp_path = None
        try:
            temp_path = os.path.join(self.upload_folder, f"temp_{uuid.uuid4()}.wav")
            file.save(temp_path)
            
            audio, sr = sf.read(temp_path)
            if len(audio) == 0:
                raise ValueError("Empty audio file")
            
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            if sr != Config.AUDIO_SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=Config.AUDIO_SAMPLE_RATE)
            
            max_samples = Config.MAX_AUDIO_DURATION * Config.AUDIO_SAMPLE_RATE
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            return audio
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file: {str(e)}")

# ==================== Audio Processor ====================
class AudioProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
    
    def process_with_timeout(self, func, *args, **kwargs):
        future = self.executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=Config.PROCESSING_TIMEOUT)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError("Processing timed out")
    
    def extract_features(self, audio):
        try:
            audio = audio / np.max(np.abs(audio))
            features = self.process_with_timeout(self._extract_features_safe, audio)
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise
    
    def _extract_features_safe(self, audio):
        """Extract audio features with error handling"""
        try:
            # Pitch extraction
            f0, _ = pw.harvest(audio.astype(np.float64), self.sample_rate)
            f0[f0 == 0] = np.nan
            f0_norm = (f0 - np.nanmean(f0)) / np.nanstd(f0)
            f0_norm = np.nan_to_num(f0_norm)
            
            # MFCC and Chroma
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
            
            return {
                'f0': f0_norm,
                'mfcc': np.mean(mfcc, axis=1),
                'chroma': np.mean(chroma, axis=1),
                'spectral_contrast': librosa.feature.spectral_contrast(
                    y=audio, sr=self.sample_rate).mean(axis=1)
            }
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            raise
    
    def calculate_similarity(self, ref_features, student_features):
        try:
            weights = {'f0': 0.4, 'mfcc': 0.4, 'chroma': 0.2}
            similarities = []
            
            # F0 similarity
            if len(ref_features['f0']) > 0 and len(student_features['f0']) > 0:
                min_len = min(len(ref_features['f0']), len(student_features['f0']))
                f0_corr = np.corrcoef(ref_features['f0'][:min_len], student_features['f0'][:min_len])[0, 1]
                similarities.append((max(f0_corr, 0), weights['f0']))
            
            # MFCC and Chroma similarities
            mfcc_sim = cosine_similarity([ref_features['mfcc']], [student_features['mfcc']])[0][0]
            chroma_sim = cosine_similarity([ref_features['chroma']], [student_features['chroma']])[0][0]
            
            similarities.extend([
                (mfcc_sim, weights['mfcc']),
                (chroma_sim, weights['chroma'])
            ])
            
            total_weight = sum(w for _, w in similarities)
            weighted_sum = sum(s * w for s, w in similarities)
            
            return max(0, min(1, weighted_sum / total_weight))
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return 0

# ==================== Application Setup ====================
audio_storage = AudioStorage(Config.UPLOAD_FOLDER)
audio_processor = AudioProcessor()

def handle_shutdown(signum, frame):
    logger.info("Shutting down gracefully...")
    audio_processor.executor.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

def memory_cleanup(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        finally:
            gc.collect()
    return decorated_function

# ==================== API Endpoints ====================
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "running",
        "service": "audio-comparison",
        "version": "1.1.0",
        "endpoints": {
            "health_check": "/health (GET)",
            "upload_reference": "/upload_reference (POST)",
            "compare_audio": "/compare_audio (POST)"
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    try:
        process = psutil.Process(os.getpid())
        return jsonify({
            "status": "healthy",
            "references": len(audio_storage.references),
            "memory_usage_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "cpu_percent": process.cpu_percent(),
            "active_threads": threading.active_count()
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/upload_reference", methods=["POST"])
@limiter.limit("3 per minute")
@memory_cleanup
def upload_reference():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    teacher_id = request.form.get("teacher_id")
    if not teacher_id or not teacher_id.isdigit():
        return jsonify({"error": "Invalid teacher_id"}), 400
        
    try:
        filepath = audio_storage.save_reference(teacher_id, request.files['audio'])
        return jsonify({
            "status": "success",
            "filepath": filepath,
            "teacher_id": teacher_id
        })
    except Exception as e:
        logger.error(f"Upload reference failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/compare_audio", methods=["POST"])
@limiter.limit("3 per minute")
@memory_cleanup
def compare_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    teacher_id = request.form.get("teacher_id")
    if not teacher_id or not teacher_id.isdigit():
        return jsonify({"error": "Invalid teacher_id"}), 400
        
    ref_path = audio_storage.get_reference_path(teacher_id)
    if not ref_path:
        return jsonify({"error": "No reference audio for this teacher"}), 404
        
    try:
        ref_audio, _ = sf.read(ref_path)
        student_audio = audio_storage._load_and_validate_audio(request.files['audio'])
        
        ref_features = audio_processor.extract_features(ref_audio)
        student_features = audio_processor.extract_features(student_audio)
        
        similarity = audio_processor.calculate_similarity(ref_features, student_features)
        
        return jsonify({
            "similarity_percent": round(similarity * 100, 2),
            "teacher_id": teacher_id,
            "features": {
                "reference": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in ref_features.items()},
                "student": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in student_features.items()}
            }
        })
    except TimeoutError:
        logger.error("Audio processing timed out")
        return jsonify({"error": "Processing timed out"}), 504
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large (max 16MB)"}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    app.run(
        host='0.0.0.0',
        port=port,
        threaded=True,
        debug=Config.DEBUG
    )

    port = int(os.getenv('PORT', 10000))
    app.run(
        host='0.0.0.0',
        port=port,
        threaded=True,
        debug=app.config['DEBUG']
    )