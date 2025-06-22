import logging
from flask import Flask, request, jsonify
import os
import uuid
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import tempfile
import datetime
import concurrent.futures
from functools import partial

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Function timed out after {timeout} seconds")

class AudioProcessor:
    def __init__(self):
        self.sr = 48000
        self.frame_length = 2048
        self.hop_length = 512
        self.max_duration = 120

    def _load_audio(self, filepath):
        try:
            try:
                audio, sr = sf.read(filepath)
            except:
                audio, sr = librosa.load(filepath, sr=self.sr)
            
            if len(audio) == 0:
                raise ValueError("Empty audio file")
            
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            if sr != self.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            
            max_samples = self.max_duration * self.sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                
            return audio
        except Exception as e:
            logger.error(f"Audio loading failed: {str(e)}")
            raise

    def _extract_formants(self, frame):
        try:
            lpc_coeffs = librosa.lpc(frame, order=10)
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))
            formants = sorted(angles * (self.sr / (2 * np.pi)))
            return formants[:5]
        except:
            return np.zeros(5)

    def extract_features(self, audio):
        try:
            if len(audio) < self.frame_length:
                raise ValueError(f"Audio too short for analysis. Length: {len(audio)} samples")
            
            audio = audio / np.max(np.abs(audio))
            
            try:
                f0, sp, ap = pw.wav2world(audio.astype(np.float64), self.sr)
            except Exception as e:
                raise ValueError(f"pyWORLD processing failed: {str(e)}")
            
            f0[f0 == 0] = np.nan
            f0_norm = (f0 - np.nanmean(f0)) / np.nanstd(f0)
            f0_norm = np.nan_to_num(f0_norm)
            
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=20)
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
            
            formants = []
            for frame in librosa.util.frame(audio, frame_length=self.frame_length, 
                                          hop_length=self.hop_length):
                if len(frame) < self.frame_length:
                    frame = np.pad(frame, (0, self.frame_length - len(frame)))
                formants.append(self._extract_formants(frame))
            
            return {
                'f0': f0_norm,
                'mfcc': np.mean(mfcc, axis=1),
                'chroma': np.mean(chroma, axis=1),
                'formants': np.mean(formants, axis=0) if formants else np.zeros(5),
                'spectral_contrast': librosa.feature.spectral_contrast(y=audio, sr=self.sr).mean(axis=1)
            }
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def calculate_similarity(self, ref_features, student_features):
        try:
            weights = {
                'f0': 0.3,
                'mfcc': 0.4,
                'chroma': 0.2,
                'formants': 0.1
            }
            
            similarities = []
            
            if len(ref_features['f0']) > 0 and len(student_features['f0']) > 0:
                min_len = min(len(ref_features['f0']), len(student_features['f0']))
                f0_corr = np.corrcoef(ref_features['f0'][:min_len], 
                                     student_features['f0'][:min_len])[0, 1]
                similarities.append((max(f0_corr, 0), weights['f0']))
            
            mfcc_sim = cosine_similarity(
                [ref_features['mfcc']], 
                [student_features['mfcc']]
            )[0][0]
            similarities.append((mfcc_sim, weights['mfcc']))
            
            chroma_sim = cosine_similarity(
                [ref_features['chroma']], 
                [student_features['chroma']]
            )[0][0]
            similarities.append((chroma_sim, weights['chroma']))
            
            formant_sim = 1 - (np.linalg.norm(
                ref_features['formants'] - student_features['formants']) / 1000)
            similarities.append((max(formant_sim, 0), weights['formants']))
            
            total_weight = sum(w for _, w in similarities)
            weighted_sum = sum(s * w for s, w in similarities)
            
            return max(0, min(1, weighted_sum / total_weight))
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return 0

class AudioStorage:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        try:
            os.makedirs(upload_folder, exist_ok=True)
            test_file = os.path.join(upload_folder, 'test_permissions')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            logger.error(f"Failed to initialize storage: {str(e)}")
            raise
        self.references = {}
    
    def get_reference_path(self, teacher_id):
        path = self.references.get(str(teacher_id))
        if path and not os.path.exists(path):
            logger.error(f"Reference file not found: {path}")
            return None
        return path
    
    def save_reference(self, teacher_id, file):
        try:
            if teacher_id in self.references:
                self._safe_remove(self.references[teacher_id])
            
            filename = f"ref_{teacher_id}_{uuid.uuid4()}.wav"
            filepath = os.path.join(self.upload_folder, filename)
            
            temp_path = self._save_temp_file(file)
            audio = self._validate_audio(temp_path)
            sf.write(filepath, audio, 48000)
            
            self.references[teacher_id] = filepath
            return filepath
        except Exception as e:
            self._safe_remove(temp_path)
            raise

    def _save_temp_file(self, file):
        fd, temp_path = tempfile.mkstemp(suffix='.ogg')
        os.close(fd)
        file.save(temp_path)
        return temp_path

    def _validate_audio(self, filepath):
        try:
            if not os.path.exists(filepath):
                raise ValueError("File does not exist")
            
            if os.path.getsize(filepath) == 0:
                raise ValueError("Empty audio file")
                
            try:
                audio, sr = sf.read(filepath)
            except:
                audio, sr = librosa.load(filepath, sr=48000)
                
            if len(audio) == 0:
                raise ValueError("Empty audio data")
                
            duration = len(audio) / sr
            if duration < 0.5:
                raise ValueError("Audio too short (min 0.5 seconds)")
                
            return audio
        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            raise

    def _safe_remove(self, filepath):
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.warning(f"Could not remove file {filepath}: {str(e)}")

app = Flask(__name__)
CORS(app, resources={
    r"/upload_reference": {"origins": "*"},
    r"/compare_audio": {"origins": "*"},
    r"/health": {"origins": "*"}
})

app.config.update(
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SECRET_KEY=os.getenv('SECRET_KEY', 'dev-key'),
    DEBUG=os.getenv('DEBUG', 'false').lower() == 'true'
)

audio_storage = AudioStorage(app.config['UPLOAD_FOLDER'])
audio_processor = AudioProcessor()

def api_response(data=None, status="success", message="", status_code=200):
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    if data is not None:
        response["data"] = data
    return jsonify(response), status_code

@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url}")

@app.route("/", methods=["GET"])
def index():
    return api_response(
        message="Audio comparison server is running",
        data={
            "endpoints": {
                "health_check": "/health (GET)",
                "upload_reference": "/upload_reference (POST)",
                "compare_audio": "/compare_audio (POST)"
            },
            "status": "operational"
        }
    )

@app.route("/health", methods=["GET"])
def health_check():
    return api_response(
        data={
            "references_count": len(audio_storage.references),
            "system": "operational"
        },
        message="Service is healthy"
    )

@app.route("/test", methods=["GET"])
def test():
    return api_response(message="Test successful")

@app.route("/upload_reference", methods=["POST", "OPTIONS"])
def upload_reference():
    if request.method == "OPTIONS":
        return api_response(message="OK")
        
    logger.info(f"Upload reference request from {request.remote_addr}")
    
    if 'audio' not in request.files:
        return api_response(
            status="error",
            message="No audio file provided",
            status_code=400
        )
        
    teacher_id = request.form.get("teacher_id")
    if not teacher_id or not teacher_id.isdigit():
        return api_response(
            status="error",
            message="Invalid teacher_id",
            status_code=400
        )
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return api_response(
            status="error",
            message="No selected file",
            status_code=400
        )

    try:
        filepath = audio_storage.save_reference(teacher_id, audio_file)
        logger.info(f"Reference audio saved for teacher {teacher_id} at {filepath}")
        return api_response(
            data={"filepath": filepath},
            message="Reference audio uploaded successfully"
        )
    except Exception as e:
        logger.error(f"Reference upload failed: {str(e)}")
        return api_response(
            status="error",
            message=str(e),
            status_code=500
        )

@app.route("/compare_audio", methods=["POST", "OPTIONS"])
def compare_audio():
    if request.method == "OPTIONS":
        return api_response(message="OK")
        
    logger.info(f"Compare audio request from {request.remote_addr}")
    
    if 'audio' not in request.files:
        return api_response(
            status="error",
            message="No audio file provided",
            status_code=400
        )
            
    teacher_id = request.form.get("teacher_id")
    if not teacher_id or not teacher_id.isdigit():
        return api_response(
            status="error",
            message="Invalid teacher_id",
            status_code=400
        )
            
    ref_path = audio_storage.get_reference_path(teacher_id)
    if not ref_path:
        return api_response(
            status="error",
            message="No reference audio for this teacher",
            status_code=404
        )
            
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return api_response(
            status="error",
            message="No selected file",
            status_code=400
        )

    temp_path = None
    try:
        temp_path = audio_storage._save_temp_file(audio_file)
        logger.info(f"Saved temp file at {temp_path}")
        
        logger.info("Loading audio files...")
        ref_audio = audio_processor._load_audio(ref_path)
        student_audio = audio_processor._load_audio(temp_path)
        
        logger.info("Extracting features...")
        try:
            ref_features = run_with_timeout(
                audio_processor.extract_features,
                args=(ref_audio,),
                timeout=30
            )
            student_features = run_with_timeout(
                audio_processor.extract_features,
                args=(student_audio,),
                timeout=30
            )
        except TimeoutError:
            logger.error("Feature extraction timed out")
            return api_response(
                status="error",
                message="Processing timed out",
                status_code=504
            )
        
        if not ref_features or not student_features:
            raise ValueError("Feature extraction failed")
        
        logger.info("Calculating similarity...")
        similarity = audio_processor.calculate_similarity(ref_features, student_features)
        
        return api_response(
            data={
                "similarity_percent": round(similarity * 100, 2),
                "features": {
                    "reference": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in ref_features.items()},
                    "student": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in student_features.items()}
                }
            },
            message="Comparison completed successfully"
        )
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        return api_response(
            status="error",
            message=str(e),
            status_code=500
        )
    finally:
        if temp_path:
            audio_storage._safe_remove(temp_path)

@app.errorhandler(413)
def request_entity_too_large(error):
    return api_response(
        status="error",
        message="File too large (max 16MB)",
        status_code=413
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 10000)),  # Изменили 5001 на 10000
        debug=app.config['DEBUG']
    )