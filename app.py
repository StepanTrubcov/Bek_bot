import logging
import os
import sqlite3
import uuid
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
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
import re
import psutil

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
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
        self.sample_rate = 48000
        self.frame_length = 2048
        self.hop_length = 512
        self.max_duration = 600
        self.min_pitch = 80
        self.max_pitch = 500
        self.pitch_diff_threshold = 8
        self.min_pitch_similarity = 0.01
        self.min_similarity = 0.05
        self.base_contour_similarity = 0.15
        self.base_spectral_similarity = 0.4
        self.different_song_threshold = 0.2
        self.severe_pitch_diff_threshold = 20
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
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
            "Богородице Дево": {
                "variants": [
                    "богородице дево радуйся",
                    "богородица дева радуйся",
                    "радуйся богородице дево"
                ],
                "priority": 0.9,
                "strict": True
            }
        }

    def _validate_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        logger.debug("Validating audio")
        try:
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
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate, stationary=True)
            return audio
        except Exception as e:
            logger.error(f"Ошибка валидации аудио: {str(e)}")
            raise AudioProcessingError(f"Не удалось обработать аудио: {str(e)}")

    def _load_audio_file(self, filepath: str) -> np.ndarray:
        logger.debug(f"Loading audio file: {filepath}")
        try:
            audio, sr = sf.read(filepath, always_2d=False)
            return self._validate_audio(audio, sr)
        except Exception as e:
            logger.error(f"Ошибка загрузки аудио: {str(e)}")
            raise AudioProcessingError(f"Не удалось загрузить аудио: {str(e)}")

    def _extract_features(self, audio: np.ndarray) -> Dict:
        logger.debug("Extracting audio features")
        try:
            features = {
                'pitch': self._extract_pitch_features(audio),
                'temporal': self._extract_temporal_features(audio),
                'spectral': self._extract_spectral_features(audio),
                'text': None
            }
            return features
        except Exception as e:
            logger.error(f"Ошибка извлечения характеристик: {str(e)}")
            raise AudioProcessingError(f"Не удалось извлечь характеристики: {str(e)}")

    def _extract_pitch_features(self, audio: np.ndarray) -> Dict:
        logger.debug("Extracting pitch features")
        try:
            f0, sp, ap = pw.wav2world(audio.astype(np.float64), self.sample_rate)
            valid_f0 = f0[(f0 >= self.min_pitch) & (f0 <= self.max_pitch)]
            if len(valid_f0) < 10:
                logger.warning("Недостаточно данных для анализа тона")
                return {
                    'mean': 0.0, 
                    'std': 0.0, 
                    'contour': [], 
                    'stability': 0.5
                }
            smoothed = self._smooth_pitch_contour(f0)
            return {
                'mean': float(np.mean(valid_f0)),
                'std': float(np.std(valid_f0)),
                'contour': smoothed.tolist(),
                'stability': float(self._calculate_pitch_stability(valid_f0))
            }
        except Exception as e:
            logger.warning(f"Ошибка извлечения тона: {str(e)}")
            return {
                'mean': 0.0, 
                'std': 0.0, 
                'contour': [], 
                'stability': 0.5
            }

    def _smooth_pitch_contour(self, f0: np.ndarray) -> np.ndarray:
        logger.debug("Smoothing pitch contour")
        try:
            valid_indices = np.where(f0 > 0)[0]
            if len(valid_indices) < 2:
                return np.zeros_like(f0)
            interp_fn = interp1d(valid_indices, f0[valid_indices], kind='linear', fill_value='extrapolate')
            smoothed = interp_fn(np.arange(len(f0)))
            smoothed = signal.medfilt(smoothed, kernel_size=5)
            smoothed = signal.savgol_filter(smoothed, window_length=9, polyorder=2)
            return smoothed
        except Exception as e:
            logger.warning(f"Ошибка сглаживания контура тона: {str(e)}")
            return np.zeros_like(f0)

    def _calculate_pitch_stability(self, f0: np.ndarray) -> float:
        logger.debug("Calculating pitch stability")
        try:
            if len(f0) < 2:
                return 0.5
            stability = 1.0 - (0.6 * (np.std(f0) / np.mean(f0)) + 
                              0.4 * (1.0 - (np.mean(np.abs(np.diff(f0))) / np.mean(f0))))
            return max(0.1, min(1.0, stability))
        except Exception as e:
            logger.warning(f"Ошибка расчета стабильности тона: {str(e)}")
            return 0.5

    def _extract_temporal_features(self, audio: np.ndarray) -> Dict:
        logger.debug("Extracting temporal features")
        try:
            duration = len(audio) / self.sample_rate
            if duration < 2.0:
                logger.warning("Аудио слишком короткое для анализа")
                return {
                    'duration': float(duration), 
                    'tempo': 0.0, 
                    'beat_count': 0, 
                    'energy_variation': 0.5
                }
            energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                       hop_length=self.hop_length).flatten()
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, 
                                                   hop_length=self.hop_length)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, 
                                     sr=self.sample_rate)
            beats = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, 
                                             hop_length=self.hop_length, units='time')
            tempo = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) and len(tempo) > 0 else float(tempo)
            return {
                'duration': float(duration),
                'tempo': tempo,
                'beat_count': int(len(beats)),
                'energy_variation': float(np.std(energy) / (np.mean(energy) + 1e-6))
            }
        except Exception as e:
            logger.warning(f"Ошибка временного анализа: {str(e)}")
            return {
                'duration': 0.0, 
                'tempo': 0.0, 
                'beat_count': 0, 
                'energy_variation': 0.5
            }

    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        logger.debug("Extracting spectral features")
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=25, 
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
        except Exception as e:
            logger.warning(f"Ошибка спектрального анализа: {str(e)}")
            return {
                'mfcc': [0.0]*25, 
                'chroma': [0.0]*12, 
                'spectral_contrast': [0.0]*7
            }

    def _recognize_song_text(self, audio_path: str) -> Optional[str]:
        logger.debug(f"Recognizing song text from {audio_path}")
        try:
            duration = sf.info(audio_path).duration
            if duration < 3.0:
                logger.warning("Аудио слишком короткое для распознавания текста")
                return None
            with sr.AudioFile(audio_path) as source:
                self.recognizer.pause_threshold = 0.8
                self.recognizer.phrase_threshold = 0.3
                audio = self.recognizer.record(source, duration=min(30, duration))
                text = self.recognizer.recognize_google(audio, language="ru-RU")
                logger.debug(f"Recognized text: {text}")
                return text.lower()
        except (sr.UnknownValueError, sr.RequestError) as e:
            logger.warning(f"Ошибка распознавания текста: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Ошибка обработки аудио для распознавания: {str(e)}")
            return None

    def _identify_song(self, text: str) -> Optional[str]:
        logger.debug(f"Identifying song from text: {text}")
        if not text:
            return None
        norm_text = self._normalize_text(text)
        best_match = None
        best_score = 0.0
        for song_name, song_data in self.expected_texts.items():
            for variant in song_data["variants"]:
                score = max(
                    Levenshtein.ratio(norm_text, variant),
                    self._partial_match_ratio(norm_text, variant)
                ) * song_data["priority"]
                if score > best_score and score > 0.85:
                    best_score = score
                    best_match = song_name
        logger.debug(f"Best match: {best_match}, score: {best_score}")
        return best_match

    def _partial_match_ratio(self, text: str, pattern: str) -> float:
        logger.debug(f"Calculating partial match ratio for text: {text}, pattern: {pattern}")
        words_text = text.split()
        words_pattern = pattern.split()
        if not words_text or not words_pattern:
            return 0.0
        if len(words_pattern) > 3 and ' '.join(words_pattern) in text:
            return 1.0
        common_words = set(words_text) & set(words_pattern)
        return len(common_words) / max(len(words_pattern), 1)

    def _compare_text_similarity(self, ref_text: str, student_text: str) -> Tuple[float, str]:
        logger.debug(f"Comparing texts: ref={ref_text}, student={student_text}")
        if not ref_text and not student_text:
            return 1.0, ""
        elif not ref_text or not student_text:
            return 0.05, "Текст не распознан"
        if ref_text == student_text:
            return 1.0, ""
        norm_ref = self._normalize_text(ref_text)
        norm_student = self._normalize_text(student_text)
        if norm_ref == norm_student:
            return 1.0, ""
        ref_song = self._identify_song(ref_text)
        student_song = self._identify_song(student_text)
        if ref_song and student_song and ref_song != student_song:
            return 0.05, f"Разные песнопения: {ref_song} vs {student_song}"
        if (ref_song and not student_song) or (not ref_song and student_song):
            return 0.1, "Не удалось точно определить песнопение"
        direct_sim = Levenshtein.ratio(norm_ref, norm_student)
        logger.debug(f"Text similarity: {direct_sim}")
        if direct_sim > 0.95:
            return 1.0, ""
        elif direct_sim > 0.9:
            return 0.8, "Незначительные отличия"
        elif direct_sim > 0.8:
            return 0.6, "Частичное совпадение"
        elif direct_sim > 0.6:
            return 0.3, "Значительные отличия"
        else:
            return 0.1, "Текст сильно отличается"

    def _normalize_text(self, text: str) -> str:
        logger.debug(f"Normalizing text: {text}")
        if not text:
            return ""
        text = re.sub(r'[^\w\s]', '', text.lower())
        stop_words = {
            "ну", "вот", "это", "как", "так", "и", "а", "но", "да", "нет",
            "ой", "ах", "ох", "ай", "эй"
        }
        num_replace = {
            '1': 'один', '2': 'два', '3': 'три', '4': 'четыре',
            '5': 'пять', '6': 'шесть', '7': 'семь', '8': 'восемь',
            '9': 'девять', '0': 'ноль'
        }
        words = []
        for word in text.split():
            if word in stop_words:
                continue
            words.append(num_replace.get(word, word))
        return ' '.join(words).strip()

    def _compare_pitch_features(self, ref: Dict, student: Dict) -> float:
        logger.debug("Comparing pitch features")
        try:
            if not ref['contour'] or not student['contour']:
                return self.min_pitch_similarity
            ref_mean = ref['mean']
            student_mean = student['mean']
            if ref_mean < self.min_pitch or student_mean < self.min_pitch:
                return self.min_pitch_similarity
            pitch_diff = abs(ref_mean - student_mean)
            if pitch_diff > self.severe_pitch_diff_threshold:
                return 0.35
            mean_sim = max(0.01, 1.0 - (pitch_diff / self.pitch_diff_threshold))
            stability_sim = 0.25 + 0.75 * (1.0 - abs(ref['stability'] - student['stability']))
            contour_sim = self._compare_pitch_contours(ref['contour'], student['contour'])
            logger.debug(f"Pitch mean sim: {mean_sim}, stability sim: {stability_sim}, contour sim: {contour_sim}")
            return max(self.min_pitch_similarity, (0.55 * mean_sim + 0.25 * stability_sim + 0.20 * contour_sim))
        except Exception as e:
            logger.warning(f"Ошибка сравнения высоты тона: {str(e)}")
            return self.min_pitch_similarity

    def _compare_pitch_contours(self, ref_contour: list, student_contour: list) -> float:
        logger.debug("Comparing pitch contours")
        try:
            min_length = min(len(ref_contour), len(student_contour))
            if min_length < 10:
                return self.base_contour_similarity
            ref_norm = np.array(ref_contour[:min_length]) / (np.mean(ref_contour[:min_length]) + 1e-6)
            student_norm = np.array(student_contour[:min_length]) / (np.mean(student_contour[:min_length]) + 1e-6)
            contour_sim = cosine_similarity([ref_norm], [student_norm])[0][0]
            correlation = np.corrcoef(ref_norm, student_norm)[0, 1]
            if np.isnan(correlation):
                correlation = self.base_contour_similarity
            logger.debug(f"Contour similarity: {contour_sim}, correlation: {correlation}")
            return max(self.base_contour_similarity, (0.65 * contour_sim + 0.35 * correlation))
        except Exception as e:
            logger.warning(f"Ошибка сравнения контуров тона: {str(e)}")
            return self.base_contour_similarity

    def _compare_temporal_features(self, ref: Dict, student: Dict) -> float:
        logger.debug("Comparing temporal features")
        try:
            duration_diff = abs(ref['duration'] - student['duration'])
            if duration_diff > 20:
                return 0.3
            duration_sim = 1 - min(1, duration_diff / 10)
            tempo_diff = abs(ref['tempo'] - student['tempo'])
            if tempo_diff > 60:
                return 0.3
            tempo_sim = 1 - min(1, tempo_diff / 20)
            beat_diff = abs(ref['beat_count'] - student['beat_count'])
            max_beats = max(1, ref['beat_count'], student['beat_count'])
            beat_sim = 1 - min(1, beat_diff / max_beats)
            logger.debug(f"Duration sim: {duration_sim}, tempo sim: {tempo_sim}, beat sim: {beat_sim}")
            return (duration_sim * 0.3 + tempo_sim * 0.5 + beat_sim * 0.2)
        except Exception as e:
            logger.warning(f"Ошибка сравнения временных характеристик: {str(e)}")
            return 0.3

    def _compare_spectral_features(self, ref: Dict, student: Dict) -> float:
        logger.debug("Comparing spectral features")
        try:
            mfcc_weights = np.array([1.0] * 5 + [0.7] * 5 + [0.5] * 5 + [0.3] * 5 + [0.1] * 5)
            mfcc_sim = cosine_similarity(
                [np.array(ref['mfcc']) * mfcc_weights],
                [np.array(student['mfcc']) * mfcc_weights]
            )[0][0]
            chroma_sim = cosine_similarity([ref['chroma']], [student['chroma']])[0][0]
            contrast_sim = cosine_similarity(
                [ref['spectral_contrast']],
                [student['spectral_contrast']]
            )[0][0]
            logger.debug(f"MFCC sim: {mfcc_sim}, chroma sim: {chroma_sim}, contrast sim: {contrast_sim}")
            return max(
                self.base_spectral_similarity,
                (0.5 * mfcc_sim + 0.3 * chroma_sim + 0.2 * contrast_sim)
            )
        except Exception as e:
            logger.warning(f"Ошибка сравнения спектральных характеристик: {str(e)}")
            return self.base_spectral_similarity

    def _calculate_final_similarity(self, text_sim: float, pitch_sim: float,
                                  temporal_sim: float, spectral_sim: float) -> float:
        logger.debug(f"Calculating final similarity: text={text_sim}, pitch={pitch_sim}, temporal={temporal_sim}, spectral={spectral_sim}")
        try:
            if text_sim < self.different_song_threshold:
                acoustic_sim = (0.2 * pitch_sim + 0.1 * temporal_sim + 0.05 * spectral_sim)
                return min(acoustic_sim, 0.2)
            weights = {
                'text': 0.35,
                'pitch': 0.45,
                'temporal': 0.15,
                'spectral': 0.05
            }
            similarity = (
                weights['text'] * text_sim +
                weights['pitch'] * pitch_sim +
                weights['temporal'] * temporal_sim +
                weights['spectral'] * spectral_sim
            )
            return max(self.min_similarity, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"Ошибка расчета итогового сходства: {str(e)}")
            return self.min_similarity

    def _prepare_result(self, similarity: float, text_sim: float, pitch_sim: float,
                       temporal_sim: float, spectral_sim: float, text_warning: str,
                       ref_features: Dict, student_features: Dict) -> Dict:
        logger.debug("Preparing result")
        try:
            result = {
                'overall_similarity_percent': round(similarity * 100, 1),
                'text_similarity_percent': round(text_sim * 100, 1),
                'pitch_similarity_percent': round(pitch_sim * 100, 1),
                'temporal_similarity_percent': round(temporal_sim * 100, 1),
                'spectral_similarity_percent': round(spectral_sim * 100, 1),
                'text_warning': text_warning,
                'details': {
                    'duration_diff': round(abs(ref_features['temporal']['duration'] - 
                                             student_features['temporal']['duration']), 1),
                    'pitch_diff': round(abs(ref_features['pitch']['mean'] - 
                                           student_features['pitch']['mean']), 1)
                },
                'warnings': [],
                'suggestions': []
            }
            if text_sim < 0.3:
                result['warnings'].append("Текст значительно отличается (возможно другое песнопение)")
                result['suggestions'].append("Проверьте, правильное ли песнопение исполняется")
            elif text_sim < 0.6:
                result['warnings'].append("Текст частично совпадает")
                result['suggestions'].append("Обратите внимание на точность произношения текста")
            if pitch_sim < 0.25:
                result['warnings'].append("Значительное отличие высоты тона")
                result['suggestions'].append("Требуется серьезная работа над тональностью исполнения")
            elif pitch_sim < 0.50:
                result['warnings'].append("Заметное отличие высоты тона")
                result['suggestions'].append("Продолжайте работать над точной интонацией")
            elif pitch_sim < 0.75:
                result['warnings'].append("Небольшое отличие высоты тона")
                result['suggestions'].append("Уточните интонацию для более точного соответствия")
            if temporal_sim < 0.5:
                result['warnings'].append("Значительное отличие в темпе исполнения")
                result['suggestions'].append("Следите за скоростью исполнения")
            return result
        except Exception as e:
            logger.error(f"Ошибка подготовки результата: {str(e)}")
            raise AudioProcessingError("Не удалось подготовить результат")

    def _perfect_match_response(self) -> Dict:
        logger.debug("Returning perfect match response")
        return {
            'overall_similarity_percent': 100.0,
            'text_similarity_percent': 100.0,
            'pitch_similarity_percent': 100.0,
            'temporal_similarity_percent': 100.0,
            'spectral_similarity_percent': 100.0,
            'text_warning': "",
            'details': {
                'pitch_similarity': 100.0,
                'text_similarity': 100.0,
                'temporal_similarity': 100.0,
                'spectral_similarity': 100.0,
                'duration_diff': 0.0,
                'pitch_diff': 0.0
            },
            'warnings': [],
            'suggestions': []
        }

    def analyze_audio(self, audio_path: str) -> Dict:
        logger.debug(f"Analyzing audio: {audio_path}")
        try:
            audio = self._load_audio_file(audio_path)
            features = self._extract_features(audio)
            features['text'] = self._recognize_song_text(audio_path)
            return features
        except Exception as e:
            logger.error(f"Анализ аудио не удался: {str(e)}")
            raise AudioProcessingError(f"Не удалось проанализировать аудио: {str(e)}")

    def compare_recordings(self, ref_path: str, student_path: str) -> Dict:
        logger.debug(f"Comparing recordings: ref={ref_path}, student={student_path}")
        try:
            if os.path.getsize(ref_path) == os.path.getsize(student_path):
                with open(ref_path, 'rb') as f1, open(student_path, 'rb') as f2:
                    if f1.read() == f2.read():
                        return self._perfect_match_response()
            ref_features = self.analyze_audio(ref_path)
            student_features = self.analyze_audio(student_path)
            if min(ref_features['temporal']['duration'], student_features['temporal']['duration']) < 2.0:
                raise AudioProcessingError("Аудио слишком короткое для анализа (минимум 2 секунды)")
            text_sim, text_warning = self._compare_text_similarity(
                ref_features.get('text', ''),
                student_features.get('text', '')
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
        except AudioProcessingError as e:
            raise
        except Exception as e:
            logger.error(f"Сравнение не удалось: {str(e)}")
            raise AudioProcessingError(f"Не удалось сравнить аудиозаписи: {str(e)}")

class AudioStorageManager:
    def __init__(self, storage_dir: str = "audio_storage"):
        self.storage_dir = os.path.abspath(storage_dir)
        self.db_path = os.path.join(self.storage_dir, "references.db")
        self.references = {}
        self._initialize_storage()
        self._initialize_db()
        self._load_references()

    def _initialize_storage(self) -> None:
        logger.debug(f"Initializing storage at {self.storage_dir}")
        try:
            os.makedirs(self.storage_dir, mode=0o775, exist_ok=True)
            logger.info(f"Аудио хранилище инициализировано в {self.storage_dir}")
        except Exception as e:
            logger.error(f"Ошибка инициализации хранилища: {str(e)}")
            raise

    def _initialize_db(self) -> None:
        logger.debug(f"Initializing SQLite database at {self.db_path}")
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS [references] (
                        user_id TEXT PRIMARY KEY,
                        filepath TEXT NOT NULL
                    )
                """)
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.OperationalError as e:
            logger.error(f"SQLite operational error during database initialization: {str(e)}")
            raise AudioProcessingError(f"Не удалось инициализировать базу данных: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {str(e)}")
            raise AudioProcessingError(f"Не удалось инициализировать базу данных: {str(e)}")

    def _load_references(self) -> None:
        logger.debug(f"Loading references from {self.db_path}")
        try:
            self.references = {}
            conn = sqlite3.connect(self.db_path, timeout=10)
            with conn:
                cursor = conn.execute("SELECT user_id, filepath FROM [references]")
                for user_id, filepath in cursor.fetchall():
                    if os.path.exists(filepath):
                        self.references[user_id] = filepath
                    else:
                        logger.warning(f"Reference file {filepath} for user_id {user_id} not found, removing")
                        conn.execute("DELETE FROM [references] WHERE user_id = ?", (user_id,))
                conn.commit()
                logger.info(f"Loaded references: {self.references}")
        except sqlite3.OperationalError as e:
            logger.error(f"SQLite operational error during loading references: {str(e)}")
            raise AudioProcessingError(f"Не удалось загрузить референсы: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during loading references: {str(e)}")
            raise AudioProcessingError(f"Не удалось загрузить референсы: {str(e)}")

    def _save_references(self) -> None:
        logger.debug(f"Saving references to {self.db_path}")
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            with conn:
                for user_id, filepath in self.references.items():
                    conn.execute("INSERT OR REPLACE INTO [references] (user_id, filepath) VALUES (?, ?)", 
                                (user_id, filepath))
                conn.commit()
                logger.info(f"References saved to {self.db_path}")
        except sqlite3.OperationalError as e:
            logger.error(f"SQLite operational error during saving references: {str(e)}")
            raise AudioProcessingError(f"Не удалось сохранить референсы: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during saving references: {str(e)}")
            raise AudioProcessingError(f"Не удалось сохранить референсы: {str(e)}")

    def _cleanup_stale_files(self) -> None:
        logger.debug("Cleaning up stale reference files")
        try:
            valid_files = set()
            conn = sqlite3.connect(self.db_path, timeout=10)
            with conn:
                cursor = conn.execute("SELECT filepath FROM [references]")
                valid_files = set(row[0] for row in cursor.fetchall())
            for filename in os.listdir(self.storage_dir):
                if filename.startswith("ref_") and filename.endswith(".wav"):
                    filepath = os.path.join(self.storage_dir, filename)
                    if filepath not in valid_files:
                        self._safe_remove_file(filepath)
        except Exception as e:
            logger.warning(f"Ошибка очистки устаревших файлов: {str(e)}")

    def save_reference_audio(self, user_id: str, audio_file) -> str:
        logger.debug(f"Saving reference audio for user_id: {user_id}")
        try:
            user_id = str(user_id).strip()
            if not user_id:
                raise AudioProcessingError("Пустой или неверный user_id")
            if user_id in self.references:
                logger.info(f"Replacing existing reference for user_id: {user_id}")
                self._safe_remove_file(self.references[user_id])
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"ref_{user_id}_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(self.storage_dir, filename)
            temp_path = self._save_temp_file(audio_file)
            audio, sr = sf.read(temp_path)
            audio = librosa.util.normalize(audio)
            sf.write(filepath, audio, sr, subtype='PCM_16')
            os.chmod(filepath, 0o664)
            self.references[user_id] = filepath
            self._save_references()
            logger.info(f"Сохранено референсное аудио для пользователя {user_id}: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Ошибка сохранения референсного аудио: {str(e)}")
            raise
        finally:
            if 'temp_path' in locals():
                self._safe_remove_file(temp_path)

    def get_reference_audio(self, user_id: str) -> Optional[str]:
        user_id = str(user_id).strip()
        logger.debug(f"Getting reference audio for user_id: {user_id}")
        try:
            self._load_references()
            ref_path = self.references.get(user_id)
            logger.debug(f"Reference path for {user_id}: {ref_path}")
            if ref_path and os.path.exists(ref_path):
                logger.debug(f"File exists: {ref_path}")
                return ref_path
            else:
                logger.warning(f"No valid reference found for user_id: {user_id}")
                if ref_path:
                    del self.references[user_id]
                    self._save_references()
                return None
        except Exception as e:
            logger.error(f"Ошибка получения референсного аудио: {str(e)}")
            raise AudioProcessingError(f"Не удалось получить референсное аудио: {str(e)}")

    def _save_temp_file(self, audio_file) -> str:
        logger.debug(f"Saving temporary file for {audio_file.filename}")
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
                ], check=True, stderr=subprocess.PIPE)
                os.remove(orig_path)
            else:
                audio_file.save(temp_path)
            os.chmod(temp_path, 0o664)
            return temp_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка конвертации аудио: {e.stderr.decode()}")
            raise AudioProcessingError("Не удалось конвертировать аудио файл")
        except Exception as e:
            logger.error(f"Ошибка сохранения временного файла: {str(e)}")
            raise AudioProcessingError(f"Не удалось сохранить временный файл: {str(e)}")

    def _safe_remove_file(self, filepath: str) -> None:
        logger.debug(f"Removing file: {filepath}")
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Файл {filepath} успешно удален")
        except Exception as e:
            logger.warning(f"Не удалось удалить файл {filepath}: {str(e)}")

class ChurchMusicServer:
    def __init__(self):
        self.app = Flask(__name__)
        self._configure_app()
        self._setup_routes()
        self.analyzer = ChurchMusicAnalyzer()
        self.storage = AudioStorageManager()
        CORS(self.app)
        logger.info("Сервер инициализирован")

    def _configure_app(self) -> None:
        logger.debug("Configuring Flask app")
        try:
            self.app.config.update(
                MAX_CONTENT_LENGTH=32 * 1024 * 1024,
                UPLOAD_FOLDER='temp_uploads',
                SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key-123'),
                DEBUG=os.getenv('DEBUG', 'false').lower() == 'true',
                JSONIFY_PRETTYPRINT_REGULAR=True
            )
        except Exception as e:
            logger.error(f"Ошибка конфигурации Flask: {str(e)}")
            raise

    def _setup_routes(self) -> None:
        logger.debug("Setting up routes")
        try:
            self.app.add_url_rule('/', 'index', self._handle_index)
            self.app.add_url_rule('/health', 'health_check', self._handle_health_check, methods=['GET'])
            self.app.add_url_rule('/upload_reference', 'upload_reference', 
                                 self._handle_upload_reference, methods=['POST'])
            self.app.add_url_rule('/compare_audio', 'compare_audio',
                                 self._handle_compare_audio, methods=['POST'])
            self.app.register_error_handler(400, self._handle_bad_request)
            self.app.register_error_handler(404, self._handle_not_found)
            self.app.register_error_handler(413, self._handle_request_too_large)
            self.app.register_error_handler(500, self._handle_internal_error)
        except Exception as e:
            logger.error(f"Ошибка настройки маршрутов: {str(e)}")
            raise

    def _api_response(self, data: Optional[Dict] = None, status: str = "success",
                     message: str = "", status_code: int = 200) -> Tuple[Dict, int]:
        logger.debug(f"Creating API response: status={status}, message={message}, status_code={status_code}")
        try:
            response = {
                "status": status,
                "message": message,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "version": "1.5"
            }
            if data is not None:
                response["data"] = data
            return jsonify(response), status_code
        except Exception as e:
            logger.error(f"Ошибка создания API ответа: {str(e)}")
            raise

    def _handle_index(self) -> Tuple[Dict, int]:
        logger.debug("Handling index request")
        try:
            return self._api_response(
                data={
                    "service": "Church Music Comparison API",
                    "description": "Сервис для сравнения церковных песнопений",
                    "endpoints": {
                        "/health": {
                            "method": "GET",
                            "description": "Проверка работоспособности сервиса"
                        },
                        "/upload_reference": {
                            "method": "POST",
                            "description": "Загрузка референсного аудио",
                            "parameters": {
                                "teacher_id": "ID преподавателя (строка)",
                                "audio": "Аудиофайл (WAV, MP3, OGG)"
                            }
                        },
                        "/compare_audio": {
                            "method": "POST",
                            "description": "Сравнение аудио с референсом",
                            "parameters": {
                                "teacher_id": "ID преподавателя (строка)",
                                "audio": "Аудиофайл для сравнения (WAV, MP3, OGG)",
                                "simplified": "Флаг для упрощённого ответа (true/false, опционально)"
                            }
                        }
                    }
                },
                message="Сервер сравнения церковных песнопений работает"
            )
        except Exception as e:
            logger.error(f"Ошибка обработки index: {str(e)}")
            return self._api_response(status="error", message="Внутренняя ошибка сервера", status_code=500)

    def _handle_health_check(self) -> Tuple[Dict, int]:
        logger.debug("Handling health check")
        try:
            return self._api_response(
                data={
                    "status": "operational",
                    "references_count": len(self.storage.references),
                    "analyzer": "ready",
                    "storage": "available",
                    "memory_usage": f"{self._get_memory_usage()} MB"
                },
                message="Сервис работает нормально"
            )
        except Exception as e:
            logger.error(f"Ошибка проверки состояния: {str(e)}")
            return self._api_response(status="error", message="Внутренняя ошибка сервера", status_code=500)

    def _get_memory_usage(self) -> float:
        logger.debug("Getting memory usage")
        try:
            return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
        except Exception as e:
            logger.warning(f"Ошибка получения использования памяти: {str(e)}")
            return 0.0

    def _handle_upload_reference(self) -> Tuple[Dict, int]:
        logger.debug("Handling upload_reference request")
        try:
            if 'audio' not in request.files:
                logger.error("No audio file provided in request")
                return self._api_response(
                    status="error",
                    message="Не предоставлен аудиофайл",
                    status_code=400
                )
            teacher_id = request.form.get("teacher_id", "").strip()
            logger.debug(f"Teacher ID: {teacher_id}")
            if not teacher_id:
                logger.error("Invalid or missing teacher_id")
                return self._api_response(
                    status="error",
                    message="Неверный идентификатор преподавателя",
                    status_code=400
                )
            audio_file = request.files['audio']
            if audio_file.filename == '':
                logger.error("No audio file selected")
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
        except AudioProcessingError as e:
            logger.error(f"Audio processing error: {str(e)}")
            return self._api_response(
                status="error",
                message=str(e),
                status_code=400
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки референса: {str(e)}")
            return self._api_response(
                status="error",
                message="Внутренняя ошибка сервера",
                status_code=500
            )

    def _handle_compare_audio(self) -> Tuple[Dict, int]:
        logger.debug("Handling compare_audio request")
        try:
            teacher_id = request.form.get("teacher_id", "").strip()
            simplified = request.args.get("simplified", "false").lower() == "true"
            logger.info(f"Received compare_audio request with teacher_id: '{teacher_id}', simplified: {simplified}")
            logger.debug(f"Available references: {self.storage.references}")
            logger.debug(f"Files in audio_storage: {os.listdir(self.storage.storage_dir)}")
            if 'audio' not in request.files:
                logger.error("No audio file provided in request")
                return self._api_response(
                    status="error",
                    message="Не предоставлен аудиофайл",
                    status_code=400
                )
            if not teacher_id:
                logger.error("Invalid or missing teacher_id")
                return self._api_response(
                    status="error",
                    message="Неверный идентификатор преподавателя",
                    status_code=400
                )
            ref_path = self.storage.get_reference_audio(teacher_id)
            if not ref_path or not os.path.exists(ref_path):
                logger.error(f"No reference audio found or file missing for teacher_id: '{teacher_id}'")
                return self._api_response(
                    status="error",
                    message="Не найдено референсное аудио для данного преподавателя",
                    status_code=404
                )
            audio_file = request.files['audio']
            if audio_file.filename == '':
                logger.error("No audio file selected")
                return self._api_response(
                    status="error",
                    message="Не выбран файл",
                    status_code=400
                )
            temp_path = None
            try:
                temp_path = self.storage._save_temp_file(audio_file)
                result = self.analyzer.compare_recordings(ref_path, temp_path)
                logger.info(f"Comparison successful for teacher_id: '{teacher_id}'")
                if simplified:
                    return self._api_response(
                        data={
                            "status": "success",
                            "similarity": result['overall_similarity_percent'],
                            "method": "server"
                        },
                        message="Сравнение аудио выполнено успешно"
                    )
                return self._api_response(
                    data={
                        "overall_similarity_percent": result['overall_similarity_percent'],
                        "text_similarity_percent": result['text_similarity_percent'],
                        "pitch_similarity_percent": result['pitch_similarity_percent'],
                        "temporal_similarity_percent": result['temporal_similarity_percent'],
                        "spectral_similarity_percent": result['spectral_similarity_percent'],
                        "text_warning": result['text_warning'],
                        "details": result['details'],
                        "warnings": result['warnings'],
                        "suggestions": result['suggestions']
                    },
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
            logger.error(f"Ошибка сравнения аудио: {str(e)}")
            return self._api_response(
                status="error",
                message="Внутренняя ошибка сервера",
                status_code=500
            )

    def _handle_bad_request(self, error) -> Tuple[Dict, int]:
        logger.error(f"Bad request: {str(error)}")
        return self._api_response(
            status="error",
            message="Неверный запрос: " + str(error),
            status_code=400
        )

    def _handle_not_found(self, error) -> Tuple[Dict, int]:
        logger.error(f"Not found: {str(error)}")
        return self._api_response(
            status="error",
            message="Ресурс не найден",
            status_code=404
        )

    def _handle_request_too_large(self, error) -> Tuple[Dict, int]:
        logger.error(f"Request too large: {str(error)}")
        return self._api_response(
            status="error",
            message="Файл слишком большой (максимум 32MB)",
            status_code=413
        )

    def _handle_internal_error(self, error) -> Tuple[Dict, int]:
        logger.error(f"Internal server error: {str(error)}")
        return self._api_response(
            status="error",
            message="Внутренняя ошибка сервера",
            status_code=500
        )

    def run(self, host: str = '0.0.0.0', port: int = 8000, debug: bool = False):
        logger.info(f"Запуск сервера сравнения церковных песнопений на {host}:{port}")
        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Ошибка запуска сервера: {str(e)}")
            raise

def create_app():
    server = ChurchMusicServer()
    return server.app

app = create_app()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)