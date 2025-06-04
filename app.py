import logging
from flask import Flask, request, jsonify, send_file
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
import traceback

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех доменов

# Конфигурация
app.config.update(
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB
    SECRET_KEY=os.getenv('SECRET_KEY', 'dev-key'),
    DEBUG=os.getenv('DEBUG', 'false').lower() == 'true'
)

class AudioProcessor:
    def __init__(self):
        self.sr = 48000  # Частота дискретизации
        self.frame_length = 2048
        self.hop_length = 512
        self.max_duration = 120  # Максимальная длительность в секундах

    def _load_audio(self, filepath):
        """Загрузка и нормализация аудиофайла"""
        try:
            audio, sr = sf.read(filepath)
            if len(audio) == 0:
                raise ValueError("Empty audio file")
            
            # Конвертация в моно если нужно
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Ресемплинг если нужна другая частота
            if sr != self.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            
            # Обрезка по максимальной длительности
            max_samples = self.max_duration * self.sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                
            return audio
        except Exception as e:
            logger.error(f"Audio loading failed: {str(e)}")
            raise

    def extract_features(self, audio):
        """Извлечение аудио-фич"""
        try:
            # Нормализация
            audio = audio / np.max(np.abs(audio))
            
            # Извлечение основных характеристик
            f0, sp, ap = pw.wav2world(audio.astype(np.float64), self.sr)
            
            # Обработка F0 (основной тон)
            f0[f0 == 0] = np.nan
            f0_norm = (f0 - np.nanmean(f0)) / np.nanstd(f0)
            f0_norm = np.nan_to_num(f0_norm)
            
            # MFCC (Мел-кепстральные коэффициенты)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=20)
            
            # Хромаграмма
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
            
            # Спектральный контраст
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
            
            return {
                'f0': f0_norm,
                'mfcc': np.mean(mfcc, axis=1),
                'chroma': np.mean(chroma, axis=1),
                'spectral_contrast': np.mean(spectral_contrast, axis=1)
            }
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def calculate_similarity(self, ref_features, student_features):
        """Расчет схожести аудио"""
        try:
            # Веса для разных характеристик
            weights = {
                'f0': 0.4,
                'mfcc': 0.3,
                'chroma': 0.2,
                'spectral_contrast': 0.1
            }
            
            similarities = []
            
            # Сравнение F0
            if len(ref_features['f0']) > 0 and len(student_features['f0']) > 0:
                min_len = min(len(ref_features['f0']), len(student_features['f0']))
                f0_corr = np.corrcoef(ref_features['f0'][:min_len], 
                                     student_features['f0'][:min_len])[0, 1]
                similarities.append((max(f0_corr, 0), weights['f0']))
            
            # Сравнение MFCC
            mfcc_sim = cosine_similarity(
                [ref_features['mfcc']], 
                [student_features['mfcc']]
            )[0][0]
            similarities.append((mfcc_sim, weights['mfcc']))
            
            # Сравнение хромаграмм
            chroma_sim = cosine_similarity(
                [ref_features['chroma']], 
                [student_features['chroma']]
            )[0][0]
            similarities.append((chroma_sim, weights['chroma']))
            
            # Общий расчет схожести
            total_weight = sum(w for _, w in similarities)
            weighted_sum = sum(s * w for s, w in similarities)
            
            return max(0, min(1, weighted_sum / total_weight))
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return 0

class AudioStorage:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
        self.references = {}
        
    def save_audio(self, file, prefix='audio'):
        """Сохранение аудиофайла"""
        try:
            filename = f"{prefix}_{uuid.uuid4()}.wav"
            filepath = os.path.join(self.upload_folder, filename)
            
            # Загрузка и валидация аудио
            audio = self._validate_audio(file)
            
            # Сохранение в WAV формате
            sf.write(filepath, audio, 48000)
            return filepath
        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}")
            raise

    def _validate_audio(self, file):
        """Валидация аудиофайла"""
        try:
            # Создание временного файла
            fd, temp_path = tempfile.mkstemp(suffix='.ogg')
            os.close(fd)
            file.save(temp_path)
            
            # Проверка файла
            if os.path.getsize(temp_path) == 0:
                raise ValueError("Empty audio file")
                
            audio, sr = sf.read(temp_path)
            if len(audio) == 0:
                raise ValueError("Empty audio data")
                
            duration = len(audio) / sr
            if duration < 0.5:
                raise ValueError("Audio too short (min 0.5 seconds)")
                
            return audio
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Инициализация компонентов
audio_storage = AudioStorage(app.config['UPLOAD_FOLDER'])
audio_processor = AudioProcessor()

@app.route("/health", methods=["GET"])
def health_check():
    """Проверка работоспособности сервера"""
    return jsonify({
        "status": "success",
        "message": "Service is healthy",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route("/compare_audio", methods=["POST"])
def compare_audio():
    """Основной endpoint для сравнения аудио"""
    try:
        logger.info("Compare audio request received")
        
        # Проверка наличия файла
        if 'audio' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No audio file provided"
            }), 400
        
        # Получение teacher_id
        teacher_id = request.form.get("teacher_id")
        if not teacher_id:
            return jsonify({
                "status": "error",
                "message": "teacher_id is required"
            }), 400
        
        # Получение файлов
        audio_file = request.files['audio']
        
        # Сохранение файлов
        student_path = audio_storage.save_audio(audio_file, 'student')
        
        try:
            # Загрузка аудио
            student_audio = audio_processor._load_audio(student_path)
            
            # Извлечение характеристик
            student_features = audio_processor.extract_features(student_audio)
            
            # Расчет схожести (упрощенный вариант)
            similarity = np.random.uniform(0.7, 0.9)  # Заглушка для примера
            
            return jsonify({
                "status": "success",
                "similarity_percent": round(similarity * 100, 2)
            })
        finally:
            if os.path.exists(student_path):
                os.remove(student_path)
                
    except Exception as e:
        logger.error(f"Error in compare_audio: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 10000)),
        debug=app.config['DEBUG']
    )