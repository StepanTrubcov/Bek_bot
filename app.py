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
        self.max_duration = 600  # 10 минут максимум
        self.min_pitch = 80      # Минимальная частота голоса (Гц)
        self.max_pitch = 500     # Максимальная частота голоса (Гц)
        
        # Параметры для сравнения (усилены для более строгого сравнения высоты тона)
        self.pitch_diff_threshold = 8     # Уменьшено с 10 до 8 для более строгого сравнения
        self.min_pitch_similarity = 0.01  # Минимальная схожесть высоты тона
        self.min_similarity = 0.05        # Минимальная общая оценка (5%)
        self.base_contour_similarity = 0.15 # Оставлено без изменений
        self.base_spectral_similarity = 0.4 # Базовая спектральная схожесть
        self.different_song_threshold = 0.2 # Порог для определения разных песнопений
        self.severe_pitch_diff_threshold = 20 # Уменьшено с 25 до 20 для более строгого контроля
        
        # Инициализация распознавателя речи
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        
        # Расширенная база текстов песнопений
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
        """Валидация и предварительная обработка аудио"""
        if len(audio) == 0:
            raise AudioProcessingError("Пустой аудиосигнал")
        
        # Конвертация в моно, если аудио стерео
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Ресемплинг до нужной частоты
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Обрезка по максимальной длительности
        max_samples = self.max_duration * self.sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Нормализация и предварительный акцент
        audio = librosa.util.normalize(audio)
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        # Подавление шума
        try:
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate, stationary=True)
        except Exception as e:
            logger.warning(f"Подавление шума не удалось: {str(e)}")
        
        return audio

    def _load_audio_file(self, filepath: str) -> np.ndarray:
        """Загрузка аудиофайла с обработкой"""
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
            'spectral': self._extract_spectral_features(audio),
            'text': None  # Будет заполнено позже
        }
        return features

    def _extract_pitch_features(self, audio: np.ndarray) -> Dict:
        """Извлечение характеристик высоты тона с улучшенной обработкой"""
        try:
            # Анализ тона с помощью WORLD
            f0, sp, ap = pw.wav2world(audio.astype(np.float64), self.sample_rate)
            
            # Фильтрация валидных значений тона
            valid_f0 = f0[(f0 >= self.min_pitch) & (f0 <= self.max_pitch)]
            
            if len(valid_f0) < 10:
                return {
                    'mean': 0.0, 
                    'std': 0.0, 
                    'contour': [], 
                    'stability': 0.5
                }
            
            # Сглаживание контура тона
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
        """Улучшенное сглаживание контура тона"""
        valid_indices = np.where(f0 > 0)[0]
        if len(valid_indices) < 2:
            return np.zeros_like(f0)
        
        # Интерполяция и сглаживание
        interp_fn = interp1d(valid_indices, f0[valid_indices], kind='linear', fill_value='extrapolate')
        smoothed = interp_fn(np.arange(len(f0)))
        
        # Медианный фильтр + фильтр Савицкого-Голея
        smoothed = signal.medfilt(smoothed, kernel_size=5)
        smoothed = signal.savgol_filter(smoothed, window_length=9, polyorder=2)
        
        return smoothed

    def _calculate_pitch_stability(self, f0: np.ndarray) -> float:
        """Расчет стабильности тона с улучшенной формулой"""
        if len(f0) < 2:
            return 0.5
            
        # Комбинированная метрика стабильности
        stability = 1.0 - (0.6 * (np.std(f0) / np.mean(f0)) + \
                    0.4 * (1.0 - (np.mean(np.abs(np.diff(f0))) / np.mean(f0))))
        
        return max(0.1, min(1.0, stability))  # Ограничение диапазона

    def _extract_temporal_features(self, audio: np.ndarray) -> Dict:
        """Извлечение временных характеристик с улучшенной обработкой"""
        try:
            duration = len(audio) / self.sample_rate
            if duration < 2.0:
                return {
                    'duration': float(duration), 
                    'tempo': 0.0, 
                    'beat_count': 0, 
                    'energy_variation': 0.5
                }
            
            # Расчет энергии сигнала
            energy = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                      hop_length=self.hop_length).flatten()
            
            # Анализ темпа и битов
            tempo, beats = 0.0, np.array([])
            try:
                onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, 
                                                      hop_length=self.hop_length)
                tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, 
                                                   sr=self.sample_rate)
                beats = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, 
                                                hop_length=self.hop_length, units='time')
                
                if isinstance(tempo, (list, np.ndarray)):
                    tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
                else:
                    tempo = float(tempo)
            except Exception as e:
                logger.warning(f"Ошибка анализа темпа: {str(e)}")
            
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
        """Извлечение спектральных характеристик с улучшенной обработкой"""
        try:
            # MFCC с увеличенным количеством коэффициентов
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=25, 
                                      hop_length=self.hop_length, n_fft=self.frame_length)
            
            # Хроматические признаки
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate, 
                                              hop_length=self.hop_length, n_chroma=12)
            
            # Спектральный контраст
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
        """Улучшенное распознавание текста песнопения"""
        try:
            duration = sf.info(audio_path).duration
            if duration < 3.0:
                return None
                
            with sr.AudioFile(audio_path) as source:
                # Настройка параметров распознавания
                self.recognizer.pause_threshold = 0.8
                self.recognizer.phrase_threshold = 0.3
                
                audio = self.recognizer.record(source, duration=min(30, duration))
                try:
                    # Попытка распознавания с Google Web Speech API
                    text = self.recognizer.recognize_google(audio, language="ru-RU")
                    return text.lower()
                except (sr.UnknownValueError, sr.RequestError) as e:
                    logger.warning(f"Ошибка распознавания текста: {str(e)}")
                    return None
        except Exception as e:
            logger.warning(f"Ошибка обработки аудио для распознавания: {str(e)}")
            return None

    def _identify_song(self, text: str) -> Optional[str]:
        """Определение песнопения по тексту с улучшенной логикой"""
        if not text:
            return None
            
        norm_text = self._normalize_text(text)
        best_match = None
        best_score = 0.0
        
        for song_name, song_data in self.expected_texts.items():
            for variant in song_data["variants"]:
                # Используем комбинацию расстояния Левенштейна и частичного совпадения
                score = max(
                    Levenshtein.ratio(norm_text, variant),
                    self._partial_match_ratio(norm_text, variant)
                ) * song_data["priority"]
                
                if score > best_score and score > 0.85:  # Повышенный порог
                    best_score = score
                    best_match = song_name
                    
        return best_match

    def _partial_match_ratio(self, text: str, pattern: str) -> float:
        """Вычисление частичного совпадения текста"""
        words_text = text.split()
        words_pattern = pattern.split()
        
        if not words_text or not words_pattern:
            return 0.0
            
        # Проверка на полное вхождение одного из вариантов
        if len(words_pattern) > 3 and ' '.join(words_pattern) in text:
            return 1.0
            
        # Подсчет совпадающих слов
        common_words = set(words_text) & set(words_pattern)
        return len(common_words) / max(len(words_pattern), 1)

    def _compare_text_similarity(self, ref_text: str, student_text: str) -> Tuple[float, str]:
        """Строгое сравнение текстов с улучшенной логикой"""
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
        
        # Определение песнопений
        ref_song = self._identify_song(ref_text)
        student_song = self._identify_song(student_text)
        
        # Если определены разные песнопения - сильно снижаем оценку
        if ref_song and student_song and ref_song != student_song:
            return 0.05, f"Разные песнопения: {ref_song} vs {student_song}"
        
        # Если одно из песнопений не определено
        if (ref_song and not student_song) or (not ref_song and student_song):
            return 0.1, "Не удалось точно определить песнопение"
        
        # Точное сравнение текстов
        direct_sim = Levenshtein.ratio(norm_ref, norm_student)
        
        # Возвращаем оценку в зависимости от степени схожести
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
        """Улучшенная нормализация текста"""
        if not text:
            return ""
        
        # Удаление пунктуации и спецсимволов
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Список стоп-слов для церковных песнопений
        stop_words = {
            "ну", "вот", "это", "как", "так", "и", "а", "но", "да", "нет",
            "ой", "ах", "ох", "ай", "эй"
        }
        
        # Замена чисел словами
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
        """Сравнение характеристик высоты тона с более строгой логикой"""
        if not ref['contour'] or not student['contour']:
            return self.min_pitch_similarity
            
        ref_mean = ref['mean']
        student_mean = student['mean']
        
        # Проверка на валидность тона
        if ref_mean < self.min_pitch or student_mean < self.min_pitch:
            return self.min_pitch_similarity
            
        # Абсолютная разница в высоте тона
        pitch_diff = abs(ref_mean - student_mean)
        
        # Если разница в высоте тона слишком большая, ограничиваем оценку
        if pitch_diff > self.severe_pitch_diff_threshold:
            return 0.35  # Уменьшено с 0.40 до 0.35 для более строгого контроля
        
        # Рассчитываем схожесть тона с более строгой шкалой
        mean_sim = max(0.01, 1.0 - (pitch_diff / self.pitch_diff_threshold))
        stability_sim = 0.25 + 0.75 * (1.0 - abs(ref['stability'] - student['stability']))  # Увеличен вес стабильности
        contour_sim = self._compare_pitch_contours(ref['contour'], student['contour'])
        
        # Более строгие веса для сравнения (увеличен вес среднего тона)
        return max(self.min_pitch_similarity, (0.55 * mean_sim + 0.25 * stability_sim + 0.20 * contour_sim))

    def _compare_pitch_contours(self, ref_contour: list, student_contour: list) -> float:
        """Сравнение контуров тона с улучшенной логикой"""
        min_length = min(len(ref_contour), len(student_contour))
        if min_length < 10:
            return self.base_contour_similarity
            
        # Нормализация контуров
        ref_norm = np.array(ref_contour[:min_length]) / (np.mean(ref_contour[:min_length]) + 1e-6)
        student_norm = np.array(student_contour[:min_length]) / (np.mean(student_contour[:min_length]) + 1e-6)
        
        # Косинусное сходство контуров
        try:
            contour_sim = cosine_similarity([ref_norm], [student_norm])[0][0]
        except ValueError:
            contour_sim = self.base_contour_similarity
        
        # Корреляция для учета формы
        correlation = np.corrcoef(ref_norm, student_norm)[0, 1]
        if np.isnan(correlation):
            correlation = self.base_contour_similarity
        
        # Среднее между косинусным сходством и корреляцией (чуть строже)
        return max(self.base_contour_similarity, (0.65 * contour_sim + 0.35 * correlation))

    def _compare_temporal_features(self, ref: Dict, student: Dict) -> float:
        """Сравнение временных характеристик с улучшенной логикой"""
        duration_diff = abs(ref['duration'] - student['duration'])
        if duration_diff > 20:
            return 0.3
        duration_sim = 1 - min(1, duration_diff / 10)  # Проверка длительности
        
        tempo_diff = abs(ref['tempo'] - student['tempo'])
        if tempo_diff > 60:
            return 0.3
        tempo_sim = 1 - min(1, tempo_diff / 20)  # Проверка темпа
        
        beat_diff = abs(ref['beat_count'] - student['beat_count'])
        max_beats = max(1, ref['beat_count'], student['beat_count'])
        beat_sim = 1 - min(1, beat_diff / max_beats)  # Проверка битов
        
        return (duration_sim * 0.3 + tempo_sim * 0.5 + beat_sim * 0.2)

    def _compare_spectral_features(self, ref: Dict, student: Dict) -> float:
        """Сравнение спектральных характеристик с улучшенной логикой"""
        try:
            # Взвешенное сравнение MFCC (первые коэффициенты важнее)
            mfcc_weights = np.array([1.0] * 5 + [0.7] * 5 + [0.5] * 5 + [0.3] * 5 + [0.1] * 5)
            mfcc_sim = cosine_similarity(
                [np.array(ref['mfcc']) * mfcc_weights],
                [np.array(student['mfcc']) * mfcc_weights]
            )[0][0]
            
            # Сравнение хроматических признаков
            chroma_sim = cosine_similarity([ref['chroma']], [student['chroma']])[0][0]
            
            # Сравнение спектрального контраста
            contrast_sim = cosine_similarity(
                [ref['spectral_contrast']],
                [student['spectral_contrast']]
            )[0][0]
            
            # Взвешенная сумма с базовой схожестью
            return max(
                self.base_spectral_similarity,
                (0.5 * mfcc_sim + 0.3 * chroma_sim + 0.2 * contrast_sim)
            )
        except Exception as e:
            logger.warning(f"Ошибка сравнения спектральных характеристик: {str(e)}")
            return self.base_spectral_similarity

    def _calculate_final_similarity(self, text_sim: float, pitch_sim: float,
                                  temporal_sim: float, spectral_sim: float) -> float:
        """Окончательный расчет схожести с более строгим весом для высоты тона"""
        # Если тексты разных песнопений (схожесть < 20%)
        if text_sim < self.different_song_threshold:
            # Жесткое ограничение в 20% для разных песнопений
            acoustic_sim = (0.2 * pitch_sim + 0.1 * temporal_sim + 0.05 * spectral_sim)
            return min(acoustic_sim, 0.2)
        
        # Основной расчет с более строгим весом для высоты тона
        weights = {
            'text': 0.35,    # Уменьшено с 0.4
            'pitch': 0.45,   # Увеличено с 0.35 для более строгого контроля высоты
            'temporal': 0.15,
            'spectral': 0.05  # Уменьшено с 0.1
        }
        
        similarity = (
            weights['text'] * text_sim +
            weights['pitch'] * pitch_sim +
            weights['temporal'] * temporal_sim +
            weights['spectral'] * spectral_sim
        )
        
        return max(self.min_similarity, min(1.0, similarity))

    def _prepare_result(self, similarity: float, text_sim: float, pitch_sim: float,
                       temporal_sim: float, spectral_sim: float, text_warning: str,
                       ref_features: Dict, student_features: Dict) -> Dict:
        """Подготовка итогового результата с более строгими предупреждениями для высоты"""
        result = {
            'similarity_percent': round(similarity * 100, 1),
            'text_warning': text_warning,
            'details': {
                'pitch_similarity': round(pitch_sim * 100, 1),
                'text_similarity': round(text_sim * 100, 1),
                'temporal_similarity': round(temporal_sim * 100, 1),
                'spectral_similarity': round(spectral_sim * 100, 1),
                'duration_diff': round(abs(ref_features['temporal']['duration'] - 
                           student_features['temporal']['duration']), 1),
                'pitch_diff': round(abs(ref_features['pitch']['mean'] - 
                         student_features['pitch']['mean']), 1)
            },
            'warnings': [],
            'suggestions': []
        }
        
        # Текстовые предупреждения
        if text_sim < 0.3:
            result['warnings'].append("Текст значительно отличается (возможно другое песнопение)")
            result['suggestions'].append("Проверьте, правильное ли песнопение исполняется")
        elif text_sim < 0.6:
            result['warnings'].append("Текст частично совпадает")
            result['suggestions'].append("Обратите внимание на точность произношения текста")
            
        # Предупреждения по высоте тона (более строгие пороги)
        if pitch_sim < 0.25:  # Уменьшено с 0.30 до 0.25
            result['warnings'].append("Значительное отличие высоты тона")
            result['suggestions'].append("Требуется серьезная работа над тональностью исполнения")
        elif pitch_sim < 0.50:  # Уменьшено с 0.55 до 0.50
            result['warnings'].append("Заметное отличие высоты тона")
            result['suggestions'].append("Продолжайте работать над точной интонацией")
        elif pitch_sim < 0.75:  # Уменьшено с 0.8 до 0.75
            result['warnings'].append("Небольшое отличие высоты тона")
            result['suggestions'].append("Уточните интонацию для более точного соответствия")
            
        # Предупреждения по темпу
        if temporal_sim < 0.5:
            result['warnings'].append("Значительное отличие в темпе исполнения")
            result['suggestions'].append("Следите за скоростью исполнения")
            
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
            'warnings': [],
            'suggestions': []
        }

    def analyze_audio(self, audio_path: str) -> Dict:
        """Полный анализ аудио с улучшенной обработкой ошибок"""
        try:
            audio = self._load_audio_file(audio_path)
            features = self._extract_features(audio)
            features['text'] = self._recognize_song_text(audio_path)
            return features
        except Exception as e:
            logger.error(f"Анализ аудио не удался: {str(e)}")
            raise AudioProcessingError("Не удалось проанализировать аудио")

    def compare_recordings(self, ref_path: str, student_path: str) -> Dict:
        """Сравнение записей с улучшенной логикой"""
        try:
            # Проверка на идентичные файлы
            if os.path.getsize(ref_path) == os.path.getsize(student_path):
                with open(ref_path, 'rb') as f1, open(student_path, 'rb') as f2:
                    if f1.read() == f2.read():
                        return self._perfect_match_response()
            
            # Анализ обеих записей
            ref_features = self.analyze_audio(ref_path)
            student_features = self.analyze_audio(student_path)
            
            # Проверка минимальной длительности
            if min(ref_features['temporal']['duration'], student_features['temporal']['duration']) < 2.0:
                raise AudioProcessingError("Аудио слишком короткое для анализа (минимум 2 секунды)")
            
            # Сравнение текстов
            text_sim, text_warning = self._compare_text_similarity(
                ref_features.get('text'),
                student_features.get('text')
            )
            
            # Сравнение характеристик
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
            
            # Расчет итоговой схожести
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
            raise AudioProcessingError("Не удалось сравнить аудиозаписи")

class AudioStorageManager:
    """Менеджер хранения аудиофайлов с улучшенной обработкой"""
    def __init__(self, storage_dir: str = "audio_storage"):
        self.storage_dir = storage_dir
        self.references = {}
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Инициализация хранилища с обработкой ошибок"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            logger.info(f"Аудио хранилище инициализировано в {self.storage_dir}")
        except Exception as e:
            logger.error(f"Ошибка инициализации хранилища: {str(e)}")
            raise

    def save_reference_audio(self, user_id: str, audio_file) -> str:
        """Сохранение референсного аудио с улучшенной обработкой"""
        try:
            # Удаление предыдущего референса, если есть
            if user_id in self.references:
                self._safe_remove_file(self.references[user_id])
            
            # Генерация уникального имени файла
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ref_{user_id}_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(self.storage_dir, filename)
            
            # Сохранение временного файла и конвертация
            temp_path = self._save_temp_file(audio_file)
            audio, sr = sf.read(temp_path)
            
            # Нормализация перед сохранением
            audio = librosa.util.normalize(audio)
            sf.write(filepath, audio, sr, subtype='PCM_16')
            
            # Обновление ссылок
            self.references[user_id] = filepath
            logger.info(f"Сохранено референсное аудио для пользователя {user_id}: {filepath}")
            
            return filepath
        except Exception as e:
            logger.error(f"Ошибка сохранения референсного аудио: {str(e)}")
            raise

    def get_reference_audio(self, user_id: str) -> Optional[str]:
        """Получение пути к референсному аудио"""
        return self.references.get(user_id)

    def _save_temp_file(self, audio_file) -> str:
        """Сохранение временного файла с конвертацией в WAV"""
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            # Если файл не WAV - конвертируем
            if audio_file.filename and not audio_file.filename.lower().endswith('.wav'):
                orig_path = temp_path + os.path.splitext(audio_file.filename)[1]
                audio_file.save(orig_path)
                
                # Конвертация с помощью ffmpeg
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
                
            return temp_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка конвертации аудио: {e.stderr.decode()}")
            raise AudioProcessingError("Не удалось конвертировать аудио файл")
        except Exception as e:
            logger.error(f"Ошибка сохранения временного файла: {str(e)}")
            raise

    def _safe_remove_file(self, filepath: str) -> None:
        """Безопасное удаление файла с обработкой ошибок"""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Файл {filepath} успешно удален")
        except Exception as e:
            logger.warning(f"Не удалось удалить файл {filepath}: {str(e)}")

class ChurchMusicServer:
    """Сервер для сравнения церковных песнопений с улучшенной обработкой"""
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
            SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key-123'),
            DEBUG=os.getenv('DEBUG', 'false').lower() == 'true',
            JSONIFY_PRETTYPRINT_REGULAR=True
        )

    def _setup_routes(self) -> None:
        """Настройка маршрутов с улучшенной обработкой"""
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

    def _api_response(self, data: Optional[Dict] = None, status: str = "success",
                     message: str = "", status_code: int = 200) -> Tuple[Dict, int]:
        """Улучшенный формат ответа API"""
        response = {
            "status": status,
            "message": message,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "version": "1.5"
        }
        if data is not None:
            response["data"] = data
        return jsonify(response), status_code

    def _handle_index(self) -> Tuple[Dict, int]:
        """Главная страница с документацией"""
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
                            "audio": "Аудиофайл для сравнения (WAV, MP3, OGG)"
                        }
                    }
                }
            },
            message="Сервер сравнения церковных песнопений работает"
        )

    def _handle_health_check(self) -> Tuple[Dict, int]:
        """Проверка здоровья сервера с детальной информацией"""
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

    def _get_memory_usage(self) -> float:
        """Получение информации об использовании памяти"""
        try:
            import psutil
            return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
        except ImportError:
            return 0.0

    def _handle_upload_reference(self) -> Tuple[Dict, int]:
        """Обработка загрузки референсного аудио с улучшенной валидацией"""
        try:
            # Проверка наличия файла
            if 'audio' not in request.files:
                return self._api_response(
                    status="error",
                    message="Не предоставлен аудиофайл",
                    status_code=400
                )
                
            # Валидация teacher_id
            teacher_id = request.form.get("teacher_id")
            if not teacher_id or not teacher_id.strip():
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
            
            # Сохранение референсного аудио
            filepath = self.storage.save_reference_audio(teacher_id, audio_file)
            
            return self._api_response(
                data={"filepath": filepath},
                message="Референсное аудио успешно загружено"
            )
        except AudioProcessingError as e:
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
        """Обработка сравнения аудио с улучшенной валидацией"""
        try:
            # Проверка наличия файла
            if 'audio' not in request.files:
                return self._api_response(
                    status="error",
                    message="Не предоставлен аудиофайл",
                    status_code=400
                )
                
            # Валидация teacher_id
            teacher_id = request.form.get("teacher_id")
            if not teacher_id or not teacher_id.strip():
                return self._api_response(
                    status="error",
                    message="Неверный идентификатор преподавателя",
                    status_code=400
                )
                
            # Проверка наличия референса
            ref_path = self.storage.get_reference_audio(teacher_id)
            if not ref_path:
                return self._api_response(
                    status="error",
                    message="Не найдено референсное аудио для данного преподавателя",
                    status_code=404
                )
                
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return self._api_response(
                    status="error",
                    message="Не выбран файл",
                    status_code=400
                )
            
            # Сохранение временного файла и сравнение
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
        """Обработка ошибки 400"""
        return self._api_response(
            status="error",
            message="Неверный запрос: " + str(error),
            status_code=400
        )

    def _handle_not_found(self, error) -> Tuple[Dict, int]:
        """Обработка ошибки 404"""
        return self._api_response(
            status="error",
            message="Ресурс не найден",
            status_code=404
        )

    def _handle_request_too_large(self, error) -> Tuple[Dict, int]:
        """Обработка ошибки 413"""
        return self._api_response(
            status="error",
            message="Файл слишком большой (максимум 32MB)",
            status_code=413
        )

    def _handle_internal_error(self, error) -> Tuple[Dict, int]:
        """Обработка ошибки 500"""
        logger.error(f"Internal server error: {str(error)}")
        return self._api_response(
            status="error",
            message="Внутренняя ошибка сервера",
            status_code=500
        )

    def run(self, host: str = '0.0.0.0', port: int = 10000, debug: bool = False):
        """Запуск сервера с улучшенной обработкой ошибок"""
        try:
            logger.info(f"Запуск сервера сравнения церковных песнопений на {host}:{port}")
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        except Exception as e:
            logger.error(f"Ошибка запуска сервера: {str(e)}")
            raise

# Создаем и настраиваем приложение
def create_app():
    server = ChurchMusicServer()
    return server.app

# Создаем экземпляр приложения
app = create_app()

if __name__ == "__main__":
    # При запуске напрямую через Python (не через Gunicorn)
    app.run(host='0.0.0.0', port=8000, debug=False)