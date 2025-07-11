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
from typing import Dict, Optional, Tuple, Union
import noisereduce as nr
import speech_recognition as sr
from scipy import signal
from scipy.interpolate import interp1d
import Levenshtein
import subprocess
import json
import re
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("church_music_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Кастомное исключение для ошибок обработки аудио"""
    pass

class ChurchMusicAnalyzer:
    def __init__(self):
        # Параметры для анализа
        self.sample_rate = 48000
        self.frame_length = 2048
        self.hop_length = 512
        self.max_duration = 600
        self.min_pitch = 80
        self.max_pitch = 500
        
        # Инициализация распознавателя речи
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        
        # База текстов песнопений с приоритетами (расширенная)
        self.expected_texts = {
            "Господи помилуй": {
                "variants": [
                    "господи помилуй",
                    "господи помилуй нас",
                    "помилуй господи",
                    "господи помилуй мя",
                    "господи помилуй меня"
                ],
                "priority": 1.0,
                "strict": True
            },
            "Отче наш": {
                "variants": [
                    "отче наш иже еси на небесех",
                    "отче наш иже еси на небеси",
                    "отче наш на небеси",
                    "отче наш который на небесах"
                ],
                "priority": 0.9,
                "strict": True
            },
            "Символ веры": {
                "variants": [
                    "верую во единаго бога отца вседержителя",
                    "верую в единого бога отца вседержителя"
                ],
                "priority": 0.8,
                "strict": True
            },
        }

    def _validate_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Валидация и обработка аудио"""
        if len(audio) == 0:
            raise AudioProcessingError("Пустой аудиосигнал")
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        max_samples = self.max_duration * self.sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        audio = librosa.util.normalize(audio)
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        try:
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate, stationary=True)
        except Exception as e:
            logger.warning(f"Noise reduction failed: {str(e)}")
        
        return audio

    def _load_audio_file(self, filepath: str) -> np.ndarray:
        """Загрузка аудиофайла"""
        try:
            audio, sr = sf.read(filepath, always_2d=False)
            return self._validate_audio(audio, sr)
        except Exception as e:
            raise AudioProcessingError(f"Ошибка загрузки аудио: {str(e)}")

    def _extract_features(self, audio: np.ndarray) -> Dict:
        """Извлечение всех характеристик аудио"""
        features = {
            'pitch': self._extract_pitch_features(audio),
            'temporal': self._extract_temporal_features(audio),
            'spectral': self._extract_spectral_features(audio)
        }
        return features

    def _extract_pitch_features(self, audio: np.ndarray) -> Dict:
        """Извлечение характеристик высоты тона"""
        try:
            f0, sp, ap = pw.wav2world(audio.astype(np.float64), self.sample_rate)
            valid_f0 = f0[(f0 >= self.min_pitch) & (f0 <= self.max_pitch)]
            
            if len(valid_f0) < 10:
                return {'mean': 0.0, 'std': 0.0, 'contour': [], 'stability': 0.0}
            
            smoothed = self._smooth_pitch_contour(f0)
            
            return {
                'mean': float(np.mean(valid_f0)),
                'std': float(np.std(valid_f0)),
                'contour': smoothed.tolist(),
                'stability': float(self._calculate_pitch_stability(valid_f0))
            }
        except Exception:
            return {'mean': 0.0, 'std': 0.0, 'contour': [], 'stability': 0.0}

    def _smooth_pitch_contour(self, f0: np.ndarray) -> np.ndarray:
        """Сглаживание контура тона"""
        valid_indices = np.where(f0 > 0)[0]
        if len(valid_indices) < 2:
            return np.zeros_like(f0)
        
        interp_fn = interp1d(valid_indices, f0[valid_indices], kind='linear', fill_value='extrapolate')
        smoothed = interp_fn(np.arange(len(f0)))
        smoothed = signal.medfilt(smoothed, kernel_size=5)
        smoothed = signal.savgol_filter(smoothed, window_length=9, polyorder=2)
        
        return smoothed

    def _calculate_pitch_stability(self, f0: np.ndarray) -> float:
        """Расчет стабильности тона"""
        if len(f0) < 2:
            return 0.0
        return 1.0 - (0.7 * (np.std(f0) / np.mean(f0)) + 0.3 * (np.mean(np.abs(np.diff(f0))) / np.mean(f0)))

    def _extract_temporal_features(self, audio: np.ndarray) -> Dict:
        """Извлечение временных характеристик"""
        try:
            duration = len(audio) / self.sample_rate
            if duration < 2.0:
                return {'duration': float(duration), 'tempo': 0.0, 'beat_count': 0, 'energy_variation': 0.0}
            
            energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length).flatten()
            
            tempo, beats = 0.0, np.array([])
            try:
                onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=self.hop_length)
                tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=self.sample_rate)
                beats = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, hop_length=self.hop_length, units='time')
                
                if isinstance(tempo, (list, np.ndarray)):
                    tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
                else:
                    tempo = float(tempo)
            except Exception:
                pass
            
            return {
                'duration': float(duration),
                'tempo': tempo,
                'beat_count': int(len(beats)),
                'energy_variation': float(np.std(energy) / (np.mean(energy) + 1e-6))
            }
        except Exception:
            return {'duration': 0.0, 'tempo': 0.0, 'beat_count': 0, 'energy_variation': 0.0}

    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """Извлечение спектральных характеристик"""
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20, 
                                      hop_length=self.hop_length, n_fft=self.frame_length)
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate, 
                                              hop_length=self.hop_length, n_chroma=12)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate,
                                                                hop_length=self.hop_length, n_bands=6)
            
            return {
                'mfcc': np.mean(mfcc, axis=1).tolist(),
                'chroma': np.mean(chroma, axis=1).tolist(),
                'spectral_contrast': np.mean(spectral_contrast, axis=1).tolist()
            }
        except Exception:
            return {'mfcc': [0.0]*20, 'chroma': [0.0]*12, 'spectral_contrast': [0.0]*7}

    def _recognize_song_text(self, audio_path: str) -> Optional[str]:
        """Распознавание текста песнопения"""
        try:
            duration = sf.info(audio_path).duration
            if duration < 3.0:
                return None
                
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source, duration=min(30, duration))
                try:
                    text = self.recognizer.recognize_google(audio, language="ru-RU")
                    return text.lower()
                except (sr.UnknownValueError, sr.RequestError):
                    return None
        except Exception:
            return None

    def _identify_song(self, text: str) -> Optional[str]:
        """Определение песнопения по тексту"""
        if not text:
            return None
            
        norm_text = self._normalize_text(text)
        best_match = None
        best_score = 0.0
        
        for song_name, song_data in self.expected_texts.items():
            for variant in song_data["variants"]:
                score = Levenshtein.ratio(norm_text, variant) * song_data["priority"]
                if score > best_score and score > 0.7:
                    best_score = score
                    best_match = song_name
                    
        return best_match

    def _compare_text_similarity(self, ref_text: str, student_text: str) -> Tuple[float, str]:
        """Сравнение текстов с жесткими критериями для разных песнопений"""
        if not ref_text and not student_text:
            return 1.0, ""
        elif not ref_text or not student_text:
            return 0.1, "Текст не распознан"
            
        if ref_text == student_text:
            return 1.0, ""
            
        # Нормализация текстов
        norm_ref = self._normalize_text(ref_text)
        norm_student = self._normalize_text(student_text)
        
        if norm_ref == norm_student:
            return 1.0, ""
        
        # Определяем песнопения
        ref_song = self._identify_song(ref_text)
        student_song = self._identify_song(student_text)
        
        # Если песнопения разные - низкий процент схожести
        if ref_song and student_song and ref_song != student_song:
            return 0.15, f"Разные песнопения: {ref_song} vs {student_song}"
        
        # Если определили только одно песнопение
        if (ref_song and not student_song) or (not ref_song and student_song):
            return 0.15, "Возможно разные песнопения"
        
        # Общее сравнение текстов
        direct_sim = Levenshtein.ratio(norm_ref, norm_student)
        if direct_sim > 0.95:
            return 1.0, ""
        elif direct_sim > 0.9:
            return 0.8, "Незначительные отличия"
        elif direct_sim > 0.8:
            return 0.6, "Частичное совпадение"
        elif direct_sim > 0.6:
            return 0.4, "Отличается"
        else:
            return 0.1, "Сильно отличается"

    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для сравнения"""
        if not text:
            return ""
        
        text = re.sub(r'[^\w\s]', '', text.lower())
        stop_words = {"ну", "вот", "это", "как", "так", "и", "а", "но", "да", "нет"}
        words = [word for word in text.split() if word not in stop_words]
        
        num_replace = {
            '1': 'один', '2': 'два', '3': 'три', '4': 'четыре',
            '5': 'пять', '6': 'шесть', '7': 'семь', '8': 'восемь',
            '9': 'девять', '0': 'ноль'
        }
        
        normalized_words = []
        for word in words:
            if word in num_replace:
                normalized_words.append(num_replace[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words).strip()

    def analyze_audio(self, audio_path: str) -> Dict:
        """Полный анализ аудио"""
        try:
            audio = self._load_audio_file(audio_path)
            features = self._extract_features(audio)
            features['text'] = self._recognize_song_text(audio_path)
            return features
        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            raise AudioProcessingError("Не удалось проанализировать аудио")

    def compare_recordings(self, ref_path: str, student_path: str) -> Dict:
        """Сравнение записей с учетом разных песнопений"""
        try:
            if os.path.getsize(ref_path) == os.path.getsize(student_path):
                with open(ref_path, 'rb') as f1, open(student_path, 'rb') as f2:
                    if f1.read() == f2.read():
                        return self._perfect_match_response()
            
            ref_features = self.analyze_audio(ref_path)
            student_features = self.analyze_audio(student_path)
            
            if min(ref_features['temporal']['duration'], student_features['temporal']['duration']) < 2.0:
                raise AudioProcessingError("Аудио слишком короткое")
            
            text_sim, text_warning = self._compare_text_similarity(
                ref_features.get('text'),
                student_features.get('text')
            )
            
            pitch_sim = self._compare_pitch_features(
                ref_features['pitch'],
                student_features['pitch']
            )
            
            temporal_sim = self._compare_temporal_features(
                ref_features['temporal'],
                student_features['temporal']
            )
            
            spectral_sim = self._compare_spectral_features(
                ref_features['spectral'],
                student_features['spectral']
            )
            
            similarity = self._calculate_final_similarity(
                text_sim, pitch_sim, temporal_sim, spectral_sim
            )
            
            return self._prepare_result(
                similarity,
                text_sim, pitch_sim, temporal_sim, spectral_sim,
                text_warning,
                ref_features, student_features
            )
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            raise AudioProcessingError("Не удалось сравнить аудиозаписи")

    def _compare_pitch_features(self, ref: Dict, student: Dict) -> float:
        """Сравнение высоты тона"""
        if not ref['contour'] or not student['contour']:
            return 0.0
            
        mean_diff = abs(ref['mean'] - student['mean'])
        if mean_diff > 100:
            return 0.2
        elif mean_diff > 50:
            return 0.5
            
        mean_sim = 1 - min(1, mean_diff / 30)
        stability_sim = 1 - abs(ref['stability'] - student['stability'])
        
        min_length = min(len(ref['contour']), len(student['contour']))
        if min_length < 10:
            contour_sim = 0.3
        else:
            contour_sim = max(0, np.corrcoef(
                ref['contour'][:min_length],
                student['contour'][:min_length]
            )[0, 1])
        
        return (mean_sim * 0.5 + stability_sim * 0.3 + contour_sim * 0.2)

    def _compare_temporal_features(self, ref: Dict, student: Dict) -> float:
        """Сравнение временных характеристик"""
        duration_diff = abs(ref['duration'] - student['duration'])
        if duration_diff > 15:
            return 0.2
        duration_sim = 1 - min(1, duration_diff / 10)
        
        tempo_diff = abs(ref['tempo'] - student['tempo'])
        if tempo_diff > 40:
            return 0.2
        tempo_sim = 1 - min(1, tempo_diff / 20)
        
        beat_diff = abs(ref['beat_count'] - student['beat_count'])
        max_beats = max(1, ref['beat_count'], student['beat_count'])
        beat_sim = 1 - min(1, beat_diff / max_beats)
        
        return (duration_sim * 0.3 + tempo_sim * 0.5 + beat_sim * 0.2)

    def _compare_spectral_features(self, ref: Dict, student: Dict) -> float:
        """Сравнение спектральных характеристик"""
        try:
            mfcc_sim = cosine_similarity([ref['mfcc']], [student['mfcc']])[0][0]
            chroma_sim = cosine_similarity([ref['chroma']], [student['chroma']])[0][0]
            contrast_sim = cosine_similarity([ref['spectral_contrast']], [student['spectral_contrast']])[0][0]
            
            return (0.5 * mfcc_sim + 0.3 * chroma_sim + 0.2 * contrast_sim)
        except Exception:
            return 0.0

    def _calculate_final_similarity(self, text_sim: float, pitch_sim: float,
                                  temporal_sim: float, spectral_sim: float) -> float:
        """Расчет итоговой схожести с учетом разных песнопений"""
        # Если тексты разных песнопений - жестко ограничиваем схожесть
        if text_sim < 0.2:
            max_similarity = 0.2
            acoustic_sim = (0.4 * pitch_sim + 0.3 * temporal_sim + 0.3 * spectral_sim)
            return min(acoustic_sim, max_similarity)
        
        weights = {
            'text': 0.6,
            'pitch': 0.2,
            'temporal': 0.1,
            'spectral': 0.1
        }
        
        similarity = (
            weights['text'] * text_sim +
            weights['pitch'] * pitch_sim +
            weights['temporal'] * temporal_sim +
            weights['spectral'] * spectral_sim
        )
        
        if all(x > 0.97 for x in [text_sim, pitch_sim, temporal_sim, spectral_sim]):
            return 1.0
            
        return max(0, min(1, similarity))

    def _prepare_result(self, similarity: float, text_sim: float, pitch_sim: float,
                       temporal_sim: float, spectral_sim: float, text_warning: str,
                       ref_features: Dict, student_features: Dict) -> Dict:
        """Подготовка итогового результата"""
        result = {
            'similarity_percent': round(similarity * 100, 2),
            'text_warning': text_warning,
            'details': {
                'pitch_similarity': round(pitch_sim * 100, 2),
                'text_similarity': round(text_sim * 100, 2),
                'temporal_similarity': round(temporal_sim * 100, 2),
                'spectral_similarity': round(spectral_sim * 100, 2),
                'duration_diff': abs(ref_features['temporal']['duration'] - 
                               student_features['temporal']['duration']),
                'pitch_diff': abs(ref_features['pitch']['mean'] - 
                             student_features['pitch']['mean'])
            },
            'warnings': []
        }
        
        if text_sim < 0.5:
            result['warnings'].append("Текст значительно отличается")
        elif text_sim < 0.8:
            result['warnings'].append("Текст частично совпадает")
            
        if pitch_sim < 0.4:
            result['warnings'].append("Высота тона значительно отличается")
        elif pitch_sim < 0.7:
            result['warnings'].append("Высота тона частично совпадает")
            
        return result

    def _perfect_match_response(self) -> Dict:
        """Ответ для полностью совпадающих записей"""
        return {
            'similarity_percent': 100.0,
            'text_warning': "",
            'details': {
                'pitch_similarity': 100.0,
                'text_similarity': 100.0,
                'temporal_similarity': 100.0,
                'spectral_similarity': 100.0,
                'duration_diff': 0.0,
                'pitch_diff': 0.0
            },
            'warnings': []
        }

class AudioStorageManager:
    """Менеджер хранения аудиофайлов"""
    def __init__(self, storage_dir: str = "audio_storage"):
        self.storage_dir = storage_dir
        self.references = {}
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Инициализация хранилища"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to initialize storage: {str(e)}")
            raise

    def save_reference_audio(self, user_id: str, audio_file) -> str:
        """Сохранение референсного аудио"""
        try:
            if user_id in self.references:
                self._safe_remove_file(self.references[user_id])
            
            filename = f"ref_{user_id}_{uuid.uuid4()}.wav"
            filepath = os.path.join(self.storage_dir, filename)
            
            temp_path = self._save_temp_file(audio_file)
            audio, sr = sf.read(temp_path)
            sf.write(filepath, audio, sr, subtype='PCM_16')
            
            self.references[user_id] = filepath
            return filepath
        except Exception as e:
            logger.error(f"Failed to save reference audio: {str(e)}")
            raise

    def get_reference_audio(self, user_id: str) -> Optional[str]:
        """Получение референсного аудио"""
        return self.references.get(user_id)

    def _save_temp_file(self, audio_file) -> str:
        """Сохранение временного файла"""
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            if audio_file.filename and not audio_file.filename.lower().endswith('.wav'):
                orig_path = temp_path + os.path.splitext(audio_file.filename)[1]
                audio_file.save(orig_path)
                
                subprocess.run([
                    'ffmpeg', '-i', orig_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '48000',
                    '-ac', '1',
                    '-y', temp_path
                ], check=True)
                os.remove(orig_path)
            else:
                audio_file.save(temp_path)
                
            return temp_path
        except Exception as e:
            logger.error(f"Failed to save temp file: {str(e)}")
            raise

    def _safe_remove_file(self, filepath: str) -> None:
        """Безопасное удаление файла"""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.warning(f"Could not remove file {filepath}: {str(e)}")

class ChurchMusicServer:
    """Сервер для сравнения церковных песнопений"""
    def __init__(self):
        self.app = Flask(__name__)
        self._configure_app()
        self._setup_routes()
        self.analyzer = ChurchMusicAnalyzer()
        self.storage = AudioStorageManager()
        CORS(self.app)

    def _configure_app(self) -> None:
        """Конфигурация Flask приложения"""
        self.app.config.update(
            MAX_CONTENT_LENGTH=32 * 1024 * 1024,
            UPLOAD_FOLDER='temp_uploads',
            SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
            DEBUG=os.getenv('DEBUG', 'false').lower() == 'true'
        )

    def _setup_routes(self) -> None:
        """Настройка маршрутов"""
        self.app.add_url_rule('/', 'index', self._handle_index)
        self.app.add_url_rule('/health', 'health_check', self._handle_health_check)
        self.app.add_url_rule('/upload_reference', 'upload_reference', 
                             self._handle_upload_reference, methods=['POST'])
        self.app.add_url_rule('/compare_audio', 'compare_audio',
                             self._handle_compare_audio, methods=['POST'])
        self.app.register_error_handler(413, self._handle_request_too_large)

    def _api_response(self, data: Optional[Dict] = None, status: str = "success",
                     message: str = "", status_code: int = 200) -> Tuple[Dict, int]:
        """Формат ответа API"""
        response = {
            "status": status,
            "message": message,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        if data is not None:
            response["data"] = data
        return jsonify(response), status_code

    def _handle_index(self) -> Tuple[Dict, int]:
        """Главная страница"""
        return self._api_response(
            data={
                "service": "Church Music Comparison API",
                "version": "1.4",
                "endpoints": {
                    "/health": "GET - Проверка работоспособности",
                    "/upload_reference": "POST - Загрузка референсного аудио",
                    "/compare_audio": "POST - Сравнение аудио"
                }
            },
            message="Сервер сравнения церковных песнопений работает"
        )

    def _handle_health_check(self) -> Tuple[Dict, int]:
        """Проверка здоровья сервера"""
        return self._api_response(
            data={
                "status": "operational",
                "references_count": len(self.storage.references),
                "analyzer": "ready"
            },
            message="Сервис работает нормально"
        )

    def _handle_upload_reference(self) -> Tuple[Dict, int]:
        """Загрузка референсного аудио"""
        try:
            if 'audio' not in request.files:
                return self._api_response(
                    status="error",
                    message="Не предоставлен аудиофайл",
                    status_code=400
                )
                
            teacher_id = request.form.get("teacher_id")
            if not teacher_id or not teacher_id.isdigit():
                return self._api_response(
                    status="error",
                    message="Неверный идентификатор преподавателя",
                    status_code=400
                )
                
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return self._api_response(
                    status="error",
                    message="Не выбран файл",
                    status_code=400
                )
            
            filepath = self.storage.save_reference_audio(teacher_id, audio_file)
            
            return self._api_response(
                data={"filepath": filepath},
                message="Референсное аудио успешно загружено"
            )
        except Exception as e:
            logger.error(f"Reference upload failed: {str(e)}")
            return self._api_response(
                status="error",
                message=str(e),
                status_code=500
            )

    def _handle_compare_audio(self) -> Tuple[Dict, int]:
        """Сравнение аудиозаписей"""
        try:
            if 'audio' not in request.files:
                return self._api_response(
                    status="error",
                    message="Не предоставлен аудиофайл",
                    status_code=400
                )
                
            teacher_id = request.form.get("teacher_id")
            if not teacher_id or not teacher_id.isdigit():
                return self._api_response(
                    status="error",
                    message="Неверный идентификатор преподавателя",
                    status_code=400
                )
                
            ref_path = self.storage.get_reference_audio(teacher_id)
            if not ref_path:
                return self._api_response(
                    status="error",
                    message="Не найдено референсное аудио",
                    status_code=404
                )
                
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return self._api_response(
                    status="error",
                    message="Не выбран файл",
                    status_code=400
                )
            
            temp_path = None
            try:
                temp_path = self.storage._save_temp_file(audio_file)
                result = self.analyzer.compare_recordings(ref_path, temp_path)
                return self._api_response(
                    data=result,
                    message="Сравнение аудио выполнено успешно"
                )
            finally:
                if temp_path:
                    self.storage._safe_remove_file(temp_path)
        except AudioProcessingError as e:
            logger.error(f"Audio processing error: {str(e)}")
            return self._api_response(
                status="error",
                message=str(e),
                status_code=400
            )
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return self._api_response(
                status="error",
                message="Внутренняя ошибка сервера",
                status_code=500
            )

    def _handle_request_too_large(self, error) -> Tuple[Dict, int]:
        """Обработка слишком большого файла"""
        return self._api_response(
            status="error",
            message="Файл слишком большой (максимум 32MB)",
            status_code=413
        )

    def run(self, host: str = '0.0.0.0', port: int = 10000, debug: bool = False):
        """Запуск сервера"""
        try:
            logger.info(f"Starting Church Music Server on {host}:{port}")
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Server failed to start: {str(e)}")
            raise

if __name__ == "__main__":
    server = ChurchMusicServer()
    server.run()