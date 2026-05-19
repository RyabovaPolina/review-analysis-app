import re
import sys
from joblib import Parallel, delayed
import logging
from pymorphy3 import MorphAnalyzer

# ═════════════════════════════════════════════════════════════════════
# КЭШИРОВАНИЕ REGEX
# ═════════════════════════════════════════════════════════════════════

# Предкомпилированные regex - НАМНОГО быстрее
REGEX_REPEAT_CHARS = re.compile(r'([а-яёa-z])\1{2,}', re.IGNORECASE)
REGEX_URLS = re.compile(r"http\S+|www\.\S+")
REGEX_HTML = re.compile(r"<.*?>")
REGEX_EMAIL = re.compile(r"\S+@\S+")
REGEX_SPECIAL = re.compile(r"[^a-zа-яё0-9\s_]")
REGEX_SPACES = re.compile(r"\s+")
_LEMMA_CACHE = {}


# ═════════════════════════════════════════════════════════════════════
# ЛЕММАТИЗАЦИЯ
# ═════════════════════════════════════════════════════════════════════
MORPH = MorphAnalyzer()

def lemmatize_tokens(tokens):
    """
    Лемматизация токенов с кэшированием
    """

    result = []

    for token in tokens:

        # Не лемматизируем доменные признаки
        if "_" in token:
            result.append(token)
            continue

        # Кэш
        if token in _LEMMA_CACHE:
            result.append(_LEMMA_CACHE[token])
            continue

        try:
            lemma = MORPH.parse(token)[0].normal_form
        except Exception:
            lemma = token

        _LEMMA_CACHE[token] = lemma
        result.append(lemma)

    return result

# ═════════════════════════════════════════════════════════════════════
# ЭТАП 1: НОРМАЛИЗАЦИЯ ДУБЛЕЙ (ПЕРЕД ОЧИСТКОЙ)
# ═════════════════════════════════════════════════════════════════════

def normalize_repeated_chars(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Схлопываем 3+ повторов в 1 символ
    text = REGEX_REPEAT_CHARS.sub(r'\1', text)

    return text

# ═════════════════════════════════════════════════════════════════════
# ЭТАП 2: БАЗОВАЯ ОЧИСТКА
# ═════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Удалить ссылки / html / email (в одной функции)
    text = REGEX_URLS.sub(" ", text)
    text = REGEX_HTML.sub(" ", text)
    text = REGEX_EMAIL.sub(" ", text)

    # Оставить только буквы и цифры
    text = REGEX_SPECIAL.sub(" ", text)

    # Нормализация пробелов
    text = REGEX_SPACES.sub(" ", text)

    return text.strip()


# ═════════════════════════════════════════════════════════════════════
# ЭТАП 3: ТОКЕНИЗАЦИЯ
# ═════════════════════════════════════════════════════════════════════

def tokenize(text: str):
    """Токенизация: разбиение на слова, фильтр очень коротких"""
    return [t for t in text.split() if len(t) >= 2]


# ═════════════════════════════════════════════════════════════════════
# ЭТАП 4: ДОМЕННЫЙ СЛОВАРЬ
# ═════════════════════════════════════════════════════════════════════

# Позитивный сленг → стандартные признаки
POSITIVE_DOMAIN_SLANG = {
    "пушка": "товар_очень_хороший",
    "бомба": "товар_очень_хороший",
    "огонь": "товар_очень_хороший",
    "шедевр": "товар_очень_хороший",
    "фишка": "товар_особенный",
    "козырь": "товар_хороший",
    
    "конфетка": "качество_отличное",
    "прелесть": "качество_отличное",
    "красотка": "качество_отличное",
    "чудо": "качество_отличное",
    
    "копейки": "цена_очень_низкая",
    "гроши": "цена_очень_низкая",
    "даром": "цена_очень_низкая",
    
    "молния": "доставка_быстрая",
    "крыло": "доставка_быстрая",
    
    "золото": "продавец_хороший",
}

# Негативный сленг → стандартные признаки
NEGATIVE_DOMAIN_SLANG = {
    "шляпа": "товар_плохой",
    "дерьмо": "товар_плохой",
    "отстой": "товар_плохой",
    "ересь": "товар_плохой",
    
    "барахло": "качество_плохое",
    "мусор": "качество_плохое",
    "хлам": "качество_плохое",
    
    "грабёж": "цена_очень_высокая",
    "разбой": "цена_очень_высокая",
    
    "черепаха": "доставка_медленная",
    "улитка": "доставка_медленная",
    
    "зло": "обслуживание_плохое",
}

# Интернет сленг
INTERNET_SLANG = {
    "прям": "очень",
    "прикольный": "хороший",
    "клёвый": "хороший",
    "кидал": "обманул",
    "облапошил": "обманул",
    "косячок": "недостаток",
}

# Многословные доменные выражения (фразы)
DOMAIN_PHRASES = {
    "кот наплакал": "количество_мало",
    "кошка наплакала": "количество_мало",
    "три волоса": "количество_мало",
    "капля в море": "количество_мало",
    
    "с ноготь": "размер_очень_маленький",
    "с горошину": "размер_маленький",
    "с кулак": "размер_средний",
    
    "как новый": "состояние_идеальное",
    "как огонь": "качество_очень_хорошее",
    
    "за так": "цена_бесплатная",
    "задаром": "цена_очень_низкая",
    "копеечка": "цена_очень_низкая",
    
    "в мигулю": "доставка_очень_быстрая",
    "в момент": "доставка_очень_быстрая",
    "век жди": "доставка_очень_медленная",
}

# ОПТИМИЗАЦИЯ: предкомпилированные regex для фраз
_PHRASE_REGEX_CACHE = {}

def _get_phrase_regex(phrase):
    """Кэш для скомпилированных regex фраз"""
    if phrase not in _PHRASE_REGEX_CACHE:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        _PHRASE_REGEX_CACHE[phrase] = re.compile(pattern)
    return _PHRASE_REGEX_CACHE[phrase]


def apply_domain_phrases(text: str) -> str:
    """
    Заменяет многословные доменные выражения на признаки
    """
    # Сортируем по длине (длинные первыми) чтобы избежать конфликтов
    sorted_phrases = sorted(DOMAIN_PHRASES.keys(), key=len, reverse=True)
    
    for phrase in sorted_phrases:
        pattern = _get_phrase_regex(phrase)
        replacement = DOMAIN_PHRASES[phrase]
        text = pattern.sub(replacement, text)
    
    return text


def apply_domain_slang(tokens):
    """
    Заменяет однословный сленг на доменные признаки
    """
    result = []
    
    for token in tokens:
        if token in POSITIVE_DOMAIN_SLANG:
            result.append(POSITIVE_DOMAIN_SLANG[token])
        elif token in NEGATIVE_DOMAIN_SLANG:
            result.append(NEGATIVE_DOMAIN_SLANG[token])
        elif token in INTERNET_SLANG:
            result.append(INTERNET_SLANG[token])
        else:
            result.append(token)
    
    return result


# ═════════════════════════════════════════════════════════════════════
# ЭТАП 5: ОБРАБОТКА ОТРИЦАНИЙ (УМНАЯ v2)
# ═════════════════════════════════════════════════════════════════════

NEGATIONS = {"не", "нет", "ни", "никогда", "ничего", "никто"}

SKIP_NEGATION_MARKERS = {
    "и", "или", "а", "но", "в", "на", "с", "по", "за",
    "к", "до", "от", "для", "об", "ли", "бы", "же"
}

SENTIMENT_NEUTRAL = {
    "это", "что", "как", "там", "тут", "вот",
    "же", "так", "еще", "ещё", "уже"
}


def is_meaningful_word(token: str) -> bool:
    """Проверить значимость слова для привязки отрицания"""
    return (
        len(token) > 2 and
        token not in SKIP_NEGATION_MARKERS and
        token not in SENTIMENT_NEUTRAL
    )


def smart_apply_negation(tokens, window=2):
    """
    Умная обработка отрицаний с поддержкой доменных признаков
    """
    result = []
    i = 0

    while i < len(tokens):
        if tokens[i] in NEGATIONS and i + 1 < len(tokens):
            # Ищем значимое слово в окне
            found = False
            
            for j in range(1, window + 1):
                if i + j < len(tokens):
                    next_token = tokens[i + j]
                    
                    if is_meaningful_word(next_token):
                        # Присоединяем отрицание
                        result.append(f"не_{next_token}")
                        
                        # Пропускаем обработанные токены
                        i = i + j + 1
                        found = True
                        break
            
            if not found:
                result.append(tokens[i])
                i += 1
        else:
            result.append(tokens[i])
            i += 1

    return result

def merge_complex_negations(tokens):
    """
    Обработка сложных отрицаний:
    не очень дорого -> не_очень_дорого
    """

    result = []
    i = 0

    while i < len(tokens):

        # не очень X
        if (
            i + 2 < len(tokens)
            and tokens[i] == "не"
            and tokens[i + 1] == "очень"
        ):
            result.append(f"не_очень_{tokens[i + 2]}")
            i += 3
            continue

        result.append(tokens[i])
        i += 1

    return result

def merge_two_word_negations(tokens):
    """Обработка 2-словных отрицаний (не очень, совсем не)"""
    TWO_WORD_PATTERNS = {
        ("не", "очень"): "не_очень",
        ("совсем", "не"): "совсем_не",
        ("вовсе", "не"): "вовсе_не",
    }
    
    result = []
    i = 0

    while i < len(tokens) - 1:
        pair = (tokens[i], tokens[i + 1])
        
        if pair in TWO_WORD_PATTERNS:
            result.append(TWO_WORD_PATTERNS[pair])
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    if i < len(tokens):
        result.append(tokens[i])

    return result


def combine_negations(tokens):
    """Комбинированная обработка отрицаний"""
    tokens = merge_complex_negations(tokens)
    tokens = merge_two_word_negations(tokens)
    tokens = smart_apply_negation(tokens, window=2)
    return tokens


# ═════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ПРЕПРОЦЕССИНГ
# ═════════════════════════════════════════════════════════════════════

def preprocess_text(text, use_negations=True, use_domain_slang=True, normalize_repeated=True):
    # ШАГ 1: Нормализация дублей
    if normalize_repeated:
        text = normalize_repeated_chars(text)
    
    # ШАГ 2: Очистка текста
    text_clean = clean_text(text)

    if not text_clean:
        return ""

    # ШАГ 3: Применяем доменные фразы
    if use_domain_slang:
        text_clean = apply_domain_phrases(text_clean)

    # ШАГ 4: Токенизируем
    tokens = tokenize(text_clean)

    # ШАГ 5: Применяем доменный сленг
    if use_domain_slang:
        tokens = apply_domain_slang(tokens)

    # ШАГ 6: Лемматизация
    tokens = lemmatize_tokens(tokens)

    # ШАГ 6: Обрабатываем отрицания
    if use_negations:
        tokens = combine_negations(tokens)

    # Fallback
    if not tokens:
        return text_clean

    return " ".join(tokens)


# ═════════════════════════════════════════════════════════════════════
# ОБРАБОТКА КОРПУСА 
# ═════════════════════════════════════════════════════════════════════

def chunkify(lst, chunk_size):
    """Разбивает список на батчи"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def preprocess_corpus(
    texts,
    verbose=True,
    n_jobs=2, 
    use_domain_slang=True,
    normalize_repeated=True,
    batch_size=2000
):
    total = len(texts)

    # Разбиваем на батчи
    batches = list(chunkify(texts, batch_size))

    if verbose:
        print(
            f"🧩 Батчей: {len(batches)} "
            f"(batch_size={batch_size}, n_jobs={n_jobs})",
            file=sys.stderr
        )

    def process_batch(batch_idx, batch):

        results = []

        for text in batch:

            processed = preprocess_text(
                text,
                use_domain_slang=use_domain_slang,
                normalize_repeated=normalize_repeated
            )
            results.append(processed)

        if verbose:
            done = min((batch_idx + 1) * batch_size, total)

            print(
                f"✓ Обработано: {done}/{total}",
                file=sys.stderr
            )

        return results

    # Параллельная обработка БАТЧЕЙ
    processed_batches = Parallel(
        n_jobs=n_jobs,
        backend="threading" 
    )(
        delayed(process_batch)(i, batch)
        for i, batch in enumerate(batches)
    )

    # Flatten
    cleaned = [
        text
        for batch in processed_batches
        for text in batch
    ]

    return cleaned


# ═════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ И ТЕСТИРОВАНИЕ
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    test_cases = [
        ("Ооочень хорошо!", "Дублирование гласных"),
        ("Уууужассссно!!!", "Дублирование согласных"),
        ("ККККРРАСССИВО", "Много дублей"),
        
        ("Пушка товара! Ооочень класно!", "Позитив + дубли"),
        ("Шляпа полная. Век жди доставку.", "Негатив + фраза"),
        ("Кот наплакал товара за копейки", "Фразы + сленг"),
        
        ("Не очень дорого и доставка быстрая", "Цена + доставка"),
        ("Не очень хороший товар, но не дорого", "Отрицание + цена"),
        ("Товар не очень хороший, зато не дорого", "Сложное отрицание"),
        ("Совсем не шляпа! Молния доставка!", "2-словное отрицание"),
        
        ("ООООЧЕНЬ КЛАСНАЯ ПУШКА!!! Кот наплакал цены. Доставка молния!!!", "Все сразу"),
        ("Ужассся шляпа... век жди... грабёж цены", "Все сразу негатив"),
        
        ("Отличный товар, не очень дорого", "Обычное без сленга"),
        ("Ужасное качество, не рекомендую", "Обычное без сленга"),
    ]

    print("=" * 120)
    print("ДЕМОНСТРАЦИЯ: ФИНАЛЬНЫЙ ПРЕПРОЦЕССИНГ v4 (ОПТИМИЗИРОВАННЫЙ)")
    print("=" * 120)
    print("✅ Нормализация дублей гласных (предкомпилированный regex)")
    print("✅ Умная обработка отрицаний")
    print("✅ Доменный словарь для интернет-магазинов")
    print("✅ Доменные фразы (с кэшированием)")
    print("=" * 120)

    for text, description in test_cases:
        processed = preprocess_text(text)
        
        print(f"\n📝 {description}")
        print(f"   Исходный:  '{text}'")
        print(f"   После:     '{processed}'")
        
        domain_tokens = [t for t in processed.split() if "_" in t]
        if domain_tokens:
            print(f"   🏪 Домен:  {domain_tokens}")

    print("\n" + "=" * 120)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 120)