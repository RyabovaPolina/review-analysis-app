import re
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional


# ═════════════════════════════════════════════════════════════════════
# АСПЕКТНЫЙ СЛОВАРЬ ДЛЯ ПРЕПРОЦЕССИРОВАННЫХ ТЕКСТОВ
# ═════════════════════════════════════════════════════════════════════

ASPECT_LEXICON = {

    # ─────────────────────────────────────────────────────────────
    # ЕДА
    # ─────────────────────────────────────────────────────────────
    "еда": {
        "trigger_words": {
            "еда",
            "блюдо",
            "меню",
            "порция",
            "вкус",
            "вкусный",
            "не_вкусный",
            "свежий",
            "не_свежий",
            "горячий",
            "холодный",
            "аромат",
            "соус",
            "специя",
            "ингредиент",
            "пересолить",
            "недосолить",
            "сырой",
            "жирный",
            "сладкий",
            "острый",
            "готовка",
            "приготовление",
        },
        "aspects": [
            "taste",
            "freshness",
            "portion_size"
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # 👔 ПЕРСОНАЛ
    # ─────────────────────────────────────────────────────────────
    "персонал": {
        "trigger_words": {
            "персонал",
            "официант",
            "кассир",
            "бармен",
            "администратор",
            "повар",
            "сотрудник",
            "работник",
            "обслуживание",
            "сервис",
            "вежливый",
            "не_вежливый",
            "грубый",
            "хам",
            "внимательный",
            "не_внимательный",
            "приветливый",
            "любезный",
            "помощь",
            "компетентный",
            "не_компетентный",
            "быстрый",
            "медленный",
        },
        "aspects": [
            "politeness",
            "attentiveness",
            "speed",
            "competence"
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # ЦЕНА
    # ─────────────────────────────────────────────────────────────
    "цена": {
        "trigger_words": {
            "цена",
            "стоимость",
            "дорого",
            "дешево",
            "бюджетно",
            "выгодно",
            "не_выгодно",
            "скидка",
            "акция",
            "переплата",
            "завышенный",
            "адекватный",
            "не_адекватный",
            "дорогой",
            "дешевый",
        },
        "aspects": [
            "expensiveness",
            "value_for_money"
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # АТМОСФЕРА
    # ─────────────────────────────────────────────────────────────
    "атмосфера": {
        "trigger_words": {
            "атмосфера",
            "интерьер",
            "дизайн",
            "обстановка",
            "уют",
            "уютный",
            "комфорт",
            "комфортный",
            "красивый",
            "стильный",
            "современный",
            "чистота",
            "чистый",
            "грязный",
            "грязь",
            "шум",
            "шумный",
            "тихий",
            "громкий",
            "музыка",
            "освещение",
            "свет",
            "темный",
            "душно",
            "вентиляция",
        },
        "aspects": [
            "comfort",
            "cleanliness",
            "design",
            "noise_level"
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # РАСПОЛОЖЕНИЕ
    # ─────────────────────────────────────────────────────────────
    "расположение": {
        "trigger_words": {
            "расположение",
            "место",
            "район",
            "центр",
            "окраина",
            "близко",
            "далеко",
            "метро",
            "станция",
            "парковка",
            "дорога",
            "улица",
            "доступ",
            "удобно",
            "проезд",
            "пешком",
        },
        "aspects": [
            "accessibility",
            "parking",
            "pedestrian_friendly"
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # ДОСТАВКА
    # ─────────────────────────────────────────────────────────────
    "доставка": {
        "trigger_words": {
            "доставка",
            "доставить",
            "курьер",
            "быстро",
            "медленно",
            "долго",
            "ожидание",
            "ждать",
            "вовремя",
            "опоздать",
            "оперативно",
            "срок",
            "приготовление",
        },
        "aspects": [
            "delivery_speed",
            "cooking_time"
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # ВСТРЕЧА / БРОНИРОВАНИЕ
    # ─────────────────────────────────────────────────────────────
    "встреча": {
        "trigger_words": {
            "встреча",
            "встретить",
            "вход",
            "зал",
            "очередь",
            "бронирование",
            "бронь",
            "стол",
            "посадить",
            "администратор",
            "резерв",
        },
        "aspects": [
            "reception",
            "seating",
            "reservation"
        ],
    },

    # ─────────────────────────────────────────────────────────────
    # НАПИТКИ
    # ─────────────────────────────────────────────────────────────
    "напитки": {
        "trigger_words": {
            "напиток",
            "кофе",
            "чай",
            "сок",
            "вода",
            "коктейль",
            "вино",
            "пиво",
            "алкоголь",
            "молоко",
            "сливки",
            "вкусный_кофе",
            "ароматный_кофе",
            "свежий_сок",
            "холодный_кофе",
            "теплый_коктейль",
        },
        "aspects": [
            "beverage_quality"
        ],
    },
}


# ═════════════════════════════════════════════════════════════════════
# ТОНАЛЬНЫЙ СЛОВАРЬ ПО АСПЕКТАМ
# ═════════════════════════════════════════════════════════════════════

SENTIMENT_LEXICON_BY_ASPECT = {

    "еда": {
        "positive": {
            "вкусный",
            "свежий",
            "ароматный",
            "нежный",
            "сочный",
            "горячий",
            "качественный",
            "аппетитный",
        },
        "negative": {
            "не_вкусный",
            "сырой",
            "жирный",
            "холодный",
            "пересолить",
            "недосолить",
            "горелый",
            "испорченный",
        },
    },

    "персонал": {
        "positive": {
            "вежливый",
            "приветливый",
            "внимательный",
            "любезный",
            "компетентный",
            "быстрый",
            "профессиональный",
        },
        "negative": {
            "грубый",
            "хам",
            "не_вежливый",
            "не_внимательный",
            "медленный",
            "не_компетентный",
            "равнодушный",
        },
    },

    "цена": {
        "positive": {
            "дешево",
            "бюджетно",
            "выгодно",
            "адекватный",
            "доступный",
        },
        "negative": {
            "дорого",
            "дорогой",
            "завышенный",
            "переплата",
            "не_выгодно",
        },
    },

    "атмосфера": {
        "positive": {
            "уютный",
            "чистый",
            "красивый",
            "стильный",
            "комфортный",
            "тихий",
            "приятный",
        },
        "negative": {
            "грязный",
            "шумный",
            "громкий",
            "темный",
            "душно",
            "не_комфортный",
        },
    },

    "расположение": {
        "positive": {
            "близко",
            "удобно",
            "доступный",
            "центр",
        },
        "negative": {
            "далеко",
            "не_удобно",
            "окраина",
            "нет_парковка",
        },
    },

    "доставка": {
        "positive": {
            "быстро",
            "оперативно",
            "вовремя",
        },
        "negative": {
            "медленно",
            "долго",
            "опоздать",
            "ожидание",
        },
    },

    "встреча": {
        "positive": {
            "быстро",
            "без_очередь",
            "приветливо",
        },
        "negative": {
            "очередь",
            "долго",
            "игнорировать",
        },
    },

    "напитки": {
        "positive": {
            "вкусный_кофе",
            "ароматный_кофе",
            "свежий_сок",
            "качественный",
        },
        "negative": {
            "холодный_кофе",
            "теплый_коктейль",
            "кислый",
            "не_вкусный",
        },
    },
}


# ═════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ЭКСТРАКТОР АСПЕКТОВ
# ═════════════════════════════════════════════════════════════════════

class AspectExtractor:
    
    def __init__(self):
        self.aspect_lexicon = ASPECT_LEXICON
        self.sentiment_lexicon = SENTIMENT_LEXICON_BY_ASPECT
        self._prepare_regex()
    
    def _prepare_regex(self):
        """Подготавливаем регулярные выражения для быстрого поиска."""
        self.aspect_patterns = {}
        
        for aspect, data in self.aspect_lexicon.items():
            # Сортируем по длине (длинные фразы первыми) чтобы избежать конфликтов
            words = sorted(data["trigger_words"], key=len, reverse=True)
            # Создаём паттерн: \b(слово1|слово2|слово3)\b
            pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
            self.aspect_patterns[aspect] = re.compile(pattern, re.IGNORECASE)
    
    def extract_aspects(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        aspects_found = defaultdict(lambda: {"mentions": 0, "sentiments": []})
        
        # Ищем все упоминания аспектов в тексте
        for aspect, pattern in self.aspect_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                aspects_found[aspect]["mentions"] = len(matches)
                
                # Определяем тональность для этого аспекта
                sentiment_score = self._get_aspect_sentiment(text_lower, aspect)
                aspects_found[aspect]["sentiment_score"] = sentiment_score
                
                # Определяем категорию: positive/negative/neutral
                if sentiment_score > 0.3:
                    aspects_found[aspect]["sentiment"] = "positive"
                elif sentiment_score < -0.3:
                    aspects_found[aspect]["sentiment"] = "negative"
                else:
                    aspects_found[aspect]["sentiment"] = "neutral"
        
        return dict(aspects_found)
    
    def _get_aspect_sentiment(self, text: str, aspect: str) -> float:
        if aspect not in self.sentiment_lexicon:
            return 0.0
        
        sentiment_words = self.sentiment_lexicon[aspect]
        positive_words = sentiment_words.get("positive", set())
        negative_words = sentiment_words.get("negative", set())
        
        # Разбиваем текст на слова
        words = text.split()
        
        # Ищем слова аспекта
        aspect_indices = []
        for i, word in enumerate(words):
            if word in self.aspect_lexicon[aspect]["trigger_words"]:
                aspect_indices.append(i)
        
        if not aspect_indices:
            return 0.0
        
        # Для каждого найденного слова аспекта ищем слова тональности в окне
        sentiment_scores = []
        context_window = 5  # ±5 слов от аспекта
        
        for idx in aspect_indices:
            window_start = max(0, idx - context_window)
            window_end = min(len(words), idx + context_window + 1)
            window = words[window_start:window_end]
            
            # Считаем positive и negative слова в окне
            pos_count = sum(1 for w in window if w in positive_words)
            neg_count = sum(1 for w in window if w in negative_words)
            

            
            # Вычисляем скор для этого окна: (positive - negative) / (positive + negative + 1)
            score = (pos_count - neg_count) / (pos_count + neg_count + 1)
            sentiment_scores.append(score)
        
        # Возвращаем средний скор
        return np.mean(sentiment_scores) if sentiment_scores else 0.0
    
    def extract_aspects_batch(self, texts: List[str]) -> List[Dict]:
        return [self.extract_aspects(text) for text in texts]


# ═════════════════════════════════════════════════════════════════════
# АГРЕГАЦИЯ АСПЕКТОВ ПО КАТЕГОРИЯМ ТОНАЛЬНОСТИ
# ═════════════════════════════════════════════════════════════════════

def aggregate_aspects_by_sentiment(
    texts_by_sentiment: Dict[str, List[str]],
    aspect_extractor: Optional[AspectExtractor] = None
) -> Dict[str, Dict]:
    if aspect_extractor is None:
        aspect_extractor = AspectExtractor()
    
    results = defaultdict(lambda: defaultdict(lambda: {
        "count": 0,
        "sentiment_sum": 0.0,
        "mention_count": 0,
    }))
    
    # Обрабатываем тексты каждой категории
    for sentiment, texts in texts_by_sentiment.items():
        for text in texts:
            aspects = aspect_extractor.extract_aspects(text)
            
            for aspect, data in aspects.items():
                results[sentiment][aspect]["count"] += 1
                results[sentiment][aspect]["sentiment_sum"] += data.get("sentiment_score", 0)
                results[sentiment][aspect]["mention_count"] += data.get("mentions", 1)
    
    # Преобразуем в финальный формат
    final_results = {}
    for sentiment, aspects in results.items():
        final_results[sentiment] = {}
        total_texts = len(texts_by_sentiment.get(sentiment, []))
        
        for aspect, stats in aspects.items():
            count = stats["count"]
            final_results[sentiment][aspect] = {
                "mentions": count,
                "avg_sentiment": stats["sentiment_sum"] / count if count > 0 else 0,
                "coverage": count / total_texts if total_texts > 0 else 0,
            }
    
    return final_results


# ═════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Примеры текстов
    demo_texts = {
        "positive": [
            "очень вкусная еда персонал вежливый место чистое атмосфера приятная",
            "отличное обслуживание быстрая доставка хорошая цена расположение удобное",
            "свежая еда внимательный официант красивый интерьер",
        ],
        "negative": [
            "невкусная еда грубый персонал грязное место долгая доставка",
            "дорого обслуживание медленное место мрачное расположение плохое",
            "сырое блюдо халатный работник запущенный интерьер",
        ],
        "neutral": [
            "еда нормальная персонал без особенностей место так себе",
            "цена высоковата но находится удобно атмосфера не очень",
        ]
    }
    
    print("=" * 120)
    print("ASPECT-BASED SENTIMENT ANALYSIS DEMO")
    print("=" * 120)
    
    extractor = AspectExtractor()
    
    # 1. Анализ отдельных текстов
    print("\n📝 АНАЛИЗ ОТДЕЛЬНЫХ ТЕКСТОВ\n")
    for sentiment, texts in demo_texts.items():
        print(f"\n{sentiment.upper()}:")
        for i, text in enumerate(texts, 1):
            aspects = extractor.extract_aspects(text)
            print(f"  {i}. '{text}'")
            for aspect, data in sorted(aspects.items()):
                sentiment_label = data.get("sentiment", "neutral")
                score = data.get("sentiment_score", 0)
                print(f"     🏷️  {aspect:15s} → {sentiment_label:10s} (score={score:+.2f})")
    
    # 2. Агрегация
    print("\n" + "=" * 120)
    print("📊 АГРЕГИРОВАННАЯ СТАТИСТИКА ПО АСПЕКТАМ\n")
    
    aggregated = aggregate_aspects_by_sentiment(demo_texts, extractor)
    
    for sentiment, aspects in sorted(aggregated.items()):
        print(f"\n{sentiment.upper()} ОТЗЫВЫ:")
        print(f"{'Аспект':<20} {'Упоминаний':<15} {'Тональность':<20} {'Покрытие':<10}")
        print("─" * 70)
        
        for aspect, stats in sorted(
            aspects.items(),
            key=lambda x: x[1]["mentions"],
            reverse=True
        ):
            mentions = stats["mentions"]
            avg_sent = stats["avg_sentiment"]
            coverage = stats["coverage"]
            
            sent_label = "😊 Позитив" if avg_sent > 0.3 else ("😠 Негатив" if avg_sent < -0.3 else "😐 Нейтраль")
            
            print(f"{aspect:<20} {mentions:<15} {sent_label} ({avg_sent:+.2f})   {coverage*100:>6.1f}%")
    
    print("\n" + "=" * 120)