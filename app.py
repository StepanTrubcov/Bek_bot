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
        # Параметры для анализа церковного пения
        self.sample_rate = 48000
        self.frame_length = 2048
        self.hop_length = 512
        self.max_duration = 600  # 10 минут максимум
        self.min_pitch = 80      # Минимальная частота для мужского голоса (Гц)
        self.max_pitch = 500     # Максимальная частота (Гц)
        
        # Инициализация распознавателя речи
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        
        # База ожидаемых текстов песнопений
        self.expected_texts = {
            "Отче наш": "отче наш иже еси на небесех да святится имя твое",
            "Символ веры": "верую во единаго бога отца вседержителя",
            "Богородице Дево": "богородице дево радуйся благодатная марие"
        }

    def _validate_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Валидация и базовая обработка аудио"""
        if len(audio) == 0:
            raise AudioProcessingError("Пустой аудиосигнал")
        
        # Конвертация в моно если нужно
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Ресемплинг до целевой частоты
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio, 
                orig_sr=sr, 
                target_sr=self.sample_rate,
                res_type='kaiser_best'
            )
        
        # Ограничение по длительности
        max_samples = self.max_duration * self.sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Нормализация громкости
        audio = librosa.util.normalize(audio)
        
        # Подавление шума
        try:
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate)
        except Exception as e:
            logger.warning(f"Noise reduction failed: {str(e)}")
        
        return audio

    def _load_audio_file(self, filepath: str) -> np.ndarray:
        """Загрузка аудиофайла с обработкой ошибок"""
        try:
            audio, sr = sf.read(filepath, always_2d=False)
            return self._validate_audio(audio, sr)
        except Exception as e:
            logger.error(f"Audio loading failed for {filepath}: {str(e)}")
            raise AudioProcessingError(f"Ошибка загрузки аудио: {str(e)}")

    def _extract_pitch_features(self, audio: np.ndarray) -> Dict:
        """Извлечение характеристик высоты тона"""
        try:
            # Анализ основного тона
            f0, sp, ap = pw.wav2world(audio.astype(np.float64), self.sample_rate)
            
            # Фильтрация невалидных значений
            valid_f0 = f0[(f0 >= self.min_pitch) & (f0 <= self.max_pitch)]
            if len(valid_f0) < 10:
                raise AudioProcessingError("Недостаточно данных о высоте тона")
            
            # Сглаживание контура
            smoothed_contour = self._smooth_pitch_contour(f0)
            
            return {
                'mean': float(np.mean(valid_f0)),
                'std': float(np.std(valid_f0)),
                'contour': smoothed_contour.tolist(),
                'stability': float(self._calculate_pitch_stability(valid_f0))
            }
        except Exception as e:
            logger.error(f"Pitch extraction failed: {str(e)}")
            raise AudioProcessingError("Ошибка анализа высоты тона")

    def _smooth_pitch_contour(self, f0: np.ndarray) -> np.ndarray:
        """Сглаживание контура основного тона"""
        valid_indices = np.where(f0 > 0)[0]
        if len(valid_indices) < 2:
            return np.zeros_like(f0)
        
        # Интерполяция для заполнения пропусков
        interp_fn = interp1d(
            valid_indices,
            f0[valid_indices],
            kind='linear',
            fill_value='extrapolate'
        )
        smoothed = interp_fn(np.arange(len(f0)))
        
        # Медианный фильтр для сглаживания
        return signal.medfilt(smoothed, kernel_size=5)

    def _calculate_pitch_stability(self, f0: np.ndarray) -> float:
        """Расчет стабильности высоты тона"""
        if len(f0) < 2:
            return 0.0
        return 1.0 - (np.std(f0) / np.mean(f0))

    def _extract_temporal_features(self, audio: np.ndarray) -> Dict:
        """Извлечение временных характеристик"""
        try:
            duration = len(audio) / self.sample_rate
            
            # Расчет энергии сигнала
            energy = librosa.feature.rms(
                y=audio,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            ).flatten()
            
            # Анализ ритма с обработкой возможных ошибок
            tempo, beats = 0.0, np.array([])
            try:
                onset_env = librosa.onset.onset_strength(
                    y=audio,
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
                tempo, beats = librosa.beat.beat_track(
                    onset_envelope=onset_env,
                    sr=self.sample_rate,
                    units='time'
                )
                
                # Обработка случая, когда beat_track возвращает None
                if tempo is None:
                    tempo = 0.0
                if beats is None:
                    beats = np.array([])
                
                # Если tempo - массив, берем первое значение
                if isinstance(tempo, np.ndarray):
                    tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
                else:
                    tempo = float(tempo)
                
            except Exception as e:
                logger.warning(f"Rhythm analysis failed: {str(e)}")
            
            return {
                'duration': float(duration),
                'tempo': tempo,
                'beat_count': int(len(beats)),
                'energy_variation': float(np.std(energy) / np.mean(energy)) if len(energy) > 0 else 0.0
            }
        except Exception as e:
            logger.error(f"Temporal features extraction failed: {str(e)}")
            raise AudioProcessingError("Ошибка анализа временных характеристик")

    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """Извлечение спектральных характеристик"""
        try:
            # MFCC с увеличенным количеством коэффициентов
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=20,
                hop_length=self.hop_length
            )
            
            # Хроматические характеристики
            chroma = librosa.feature.chroma_cqt(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Спектральные контрасты
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            return {
                'mfcc': np.mean(mfcc, axis=1).tolist(),
                'chroma': np.mean(chroma, axis=1).tolist(),
                'spectral_contrast': np.mean(spectral_contrast, axis=1).tolist()
            }
        except Exception as e:
            logger.error(f"Spectral features extraction failed: {str(e)}")
            raise AudioProcessingError("Ошибка анализа спектральных характеристик")

    def analyze_audio(self, audio_path: str) -> Dict:
        """Полный анализ аудиофайла"""
        try:
            audio = self._load_audio_file(audio_path)
            logger.info(f"Analyzing audio file: {audio_path}")
            
            features = {
                'pitch': self._extract_pitch_features(audio),
                'temporal': self._extract_temporal_features(audio),
                'spectral': self._extract_spectral_features(audio)
            }
            
            logger.info(f"Analysis completed for {audio_path}")
            return features
        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            raise AudioProcessingError("Не удалось проанализировать аудиофайл")

    def compare_recordings(self, ref_path: str, student_path: str) -> Dict:
        """Сравнение двух аудиозаписей"""
        try:
            logger.info(f"Starting comparison: ref={ref_path}, student={student_path}")
            
            # Анализ обеих записей
            ref_features = self.analyze_audio(ref_path)
            student_features = self.analyze_audio(student_path)
            
            # Расчет схожести
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
            
            # Итоговая оценка схожести (взвешенная сумма)
            similarity = (0.5 * pitch_sim + 0.3 * temporal_sim + 0.2 * spectral_sim)
            similarity_percent = max(0, min(100, similarity * 100))
            
            result = {
                'similarity_percent': float(round(similarity_percent, 2)),
                'details': {
                    'pitch_similarity': float(round(pitch_sim * 100, 2)),
                    'temporal_similarity': float(round(temporal_sim * 100, 2)),
                    'spectral_similarity': float(round(spectral_sim * 100, 2)),
                    'duration_diff': float(abs(
                        ref_features['temporal']['duration'] - 
                        student_features['temporal']['duration']
                    ))
                }
            }
            
            logger.info(f"Comparison result: {result['similarity_percent']}%")
            return result
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}", exc_info=True)
            raise AudioProcessingError("Не удалось сравнить аудиозаписи")

    def _compare_pitch_features(self, ref: Dict, student: Dict) -> float:
        """Сравнение характеристик высоты тона"""
        try:
            # Сравнение средних значений
            mean_sim = 1 - min(1, abs(ref['mean'] - student['mean']) / 50)
            
            # Сравнение стабильности
            stability_sim = 1 - abs(ref['stability'] - student['stability'])
            
            # Сравнение контуров с помощью корреляции
            contour_sim = 0.5  # Значение по умолчанию
            if len(ref['contour']) == len(student['contour']):
                contour_sim = max(0, np.corrcoef(ref['contour'], student['contour'])[0, 1])
            
            return (mean_sim + stability_sim + contour_sim) / 3
        except Exception as e:
            logger.error(f"Pitch comparison failed: {str(e)}")
            return 0.0

    def _compare_temporal_features(self, ref: Dict, student: Dict) -> float:
        """Сравнение временных характеристик"""
        try:
            # Сравнение длительности
            duration_sim = 1 - min(1, abs(ref['duration'] - student['duration']) / 5)
            
            # Сравнение темпа
            tempo_sim = 1 - min(1, abs(ref['tempo'] - student['tempo']) / 20)
            
            # Сравнение ритмического рисунка
            beat_sim = 1 - min(1, abs(ref['beat_count'] - student['beat_count']) / 
                              max(ref['beat_count'], student['beat_count'], 1))
            
            return (duration_sim + tempo_sim + beat_sim) / 3
        except Exception as e:
            logger.error(f"Temporal comparison failed: {str(e)}")
            return 0.0

    def _compare_spectral_features(self, ref: Dict, student: Dict) -> float:
        """Сравнение спектральных характеристик"""
        try:
            # Сравнение MFCC
            mfcc_sim = cosine_similarity(
                [ref['mfcc']],
                [student['mfcc']]
            )[0][0]
            
            # Сравнение хроматических признаков
            chroma_sim = cosine_similarity(
                [ref['chroma']],
                [student['chroma']]
            )[0][0]
            
            # Сравнение спектральных контрастов
            contrast_sim = cosine_similarity(
                [ref['spectral_contrast']],
                [student['spectral_contrast']]
            )[0][0]
            
            return (mfcc_sim + chroma_sim + contrast_sim) / 3
        except Exception as e:
            logger.error(f"Spectral comparison failed: {str(e)}")
            return 0.0

class AudioStorageManager:
    """Менеджер для хранения и управления аудиофайлами"""
    def __init__(self, storage_dir: str = "audio_storage"):
        self.storage_dir = storage_dir
        self.references = {}
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Инициализация директорий для хранения"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            logger.info(f"Storage initialized at {os.path.abspath(self.storage_dir)}")
        except Exception as e:
            logger.error(f"Failed to initialize storage: {str(e)}")
            raise

    def save_reference_audio(self, user_id: str, audio_file) -> str:
        """Сохранение референсного аудио"""
        try:
            # Удаляем старый референс, если есть
            if user_id in self.references:
                self._safe_remove_file(self.references[user_id])
            
            # Создаем уникальное имя файла
            filename = f"ref_{user_id}_{uuid.uuid4()}.wav"
            filepath = os.path.join(self.storage_dir, filename)
            
            # Сохраняем временный файл и конвертируем в WAV
            temp_path = self._save_temp_file(audio_file)
            audio, sr = sf.read(temp_path)
            sf.write(filepath, audio, sr, subtype='PCM_16')
            
            # Обновляем ссылку на референс
            self.references[user_id] = filepath
            logger.info(f"Reference audio saved for user {user_id} at {filepath}")
            
            return filepath
        except Exception as e:
            logger.error(f"Failed to save reference audio: {str(e)}")
            raise

    def get_reference_audio(self, user_id: str) -> Optional[str]:
        """Получение пути к референсному аудио"""
        return self.references.get(user_id)

    def _save_temp_file(self, audio_file) -> str:
        """Сохранение временного файла"""
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.ogg')
            os.close(fd)
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
                logger.info(f"Removed file: {filepath}")
        except Exception as e:
            logger.warning(f"Could not remove file {filepath}: {str(e)}")

class ChurchMusicServer:
    """Основной класс сервера для сравнения церковных песнопений"""
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
            MAX_CONTENT_LENGTH=32 * 1024 * 1024,  # 32MB
            UPLOAD_FOLDER='temp_uploads',
            SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
            DEBUG=os.getenv('DEBUG', 'false').lower() == 'true'
        )

    def _setup_routes(self) -> None:
        """Настройка маршрутов API"""
        self.app.add_url_rule('/', 'index', self._handle_index)
        self.app.add_url_rule('/health', 'health_check', self._handle_health_check)
        self.app.add_url_rule(
            '/upload_reference', 
            'upload_reference', 
            self._handle_upload_reference, 
            methods=['POST']
        )
        self.app.add_url_rule(
            '/compare_audio', 
            'compare_audio', 
            self._handle_compare_audio, 
            methods=['POST']
        )
        self.app.register_error_handler(413, self._handle_request_too_large)

    def _api_response(
        self, 
        data: Optional[Dict] = None, 
        status: str = "success", 
        message: str = "", 
        status_code: int = 200
    ) -> Tuple[Dict, int]:
        """Унифицированный формат ответа API"""
        response = {
            "status": status,
            "message": message,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        if data is not None:
            response["data"] = data
        return jsonify(response), status_code

    def _handle_index(self) -> Tuple[Dict, int]:
        """Обработчик главной страницы"""
        return self._api_response(
            data={
                "service": "Church Music Comparison API",
                "version": "1.0",
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
            logger.error(f"Comparison failed: {str(e)}", exc_info=True)
            return self._api_response(
                status="error",
                message="Внутренняя ошибка сервера",
                status_code=500
            )

    def _handle_request_too_large(self, error) -> Tuple[Dict, int]:
        """Обработчик ошибки слишком большого файла"""
        return self._api_response(
            status="error",
            message="Файл слишком большой (максимум 32MB)",
            status_code=413
        )

    def run(self, host: str = '0.0.0.0', port: int = 10000, debug: bool = False):
        """Запуск сервера"""
        try:
            logger.info(f"Starting Church Music Server on {host}:{port}")
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Server failed to start: {str(e)}")
            raise

if __name__ == "__main__":
    server = ChurchMusicServer()
    server.run()
    # Создание и запуск сервера
    server = ChurchMusicServer()
    server.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 10000)),
        debug=os.getenv('DEBUG', 'false').lower() == 'true'
    )