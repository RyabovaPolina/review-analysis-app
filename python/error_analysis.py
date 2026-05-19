import numpy as np
from typing import List
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline


# ═════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС АНАЛИЗА ОШИБОК
# ═════════════════════════════════════════════════════════════════════

class ErrorAnalyzer:
    
    def __init__(self, model: Pipeline, X_test: List[str], y_test: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            model: обученный sklearn pipeline
            X_test: тестовые тексты
            y_test: истинные метки
            y_pred: предсказанные метки
        """
        self.model = model
        # ВАЖНО: приведение к numpy
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.y_pred = np.array(y_pred)
            
        # Вычисляем индексы ошибок
        self.errors_mask = self.y_test != self.y_pred
        self.X_errors = [self.X_test[i] for i in range(len(self.X_test)) if self.errors_mask[i]]
        self.y_true_errors = self.y_test[self.errors_mask]
        self.y_pred_errors = self.y_pred[self.errors_mask]
        
        self.error_pairs = {}

        classes = np.unique(np.concatenate([self.y_test, self.y_pred]))

        for true_class in classes:
            for pred_class in classes:

                if true_class == pred_class:
                    continue

                indices = np.where(
                    (self.y_test == true_class) &
                    (self.y_pred == pred_class)
                )[0]

                self.error_pairs[(true_class, pred_class)] = indices    
    
    # ═════════════════════════════════════════════════════════════════
    # 1. ОБЩАЯ СТАТИСТИКА
    # ═════════════════════════════════════════════════════════════════
    
    def print_summary(self):

        total = len(self.y_test)
        errors = np.sum(self.errors_mask)
        accuracy = (total - errors) / total

        print("\n" + "=" * 80)
        print("📊 СВОДКА ПО ОШИБКАМ")
        print("=" * 80)

        print(f"\n✅ Всего примеров: {total}")
        print(f"❌ Ошибок всего: {errors} ({errors/total:.1%})")
        print(f"🎯 Accuracy: {accuracy:.4f}")

        print("\n📉 CONFUSION PAIRS:")

        for (true_class, pred_class), indices in self.error_pairs.items():

            if len(indices) == 0:
                continue

            print(
                f"   {true_class} -> {pred_class}: "
                f"{len(indices)}"
            )

        print("\n" + "=" * 80)
    
    def analyze_confusion_pair(
        self,
        true_class,
        pred_class,
        top_n=5
    ):

        indices = self.error_pairs.get(
            (true_class, pred_class),
            []
        )

        if len(indices) == 0:

            print(
                f"\n✅ Ошибок "
                f"{true_class} -> {pred_class} нет"
            )

            return

        print("\n" + "=" * 80)
        print(f"⚠️  {true_class.upper()} -> {pred_class.upper()}")
        print("=" * 80)

        clf = self.model.named_steps["clf"]
        classes = list(clf.classes_)

        pred_idx = classes.index(pred_class)

        # ==========================================================
        # 1. Получаем confidence scores
        # ==========================================================

        if hasattr(clf, "predict_proba"):

            proba = self.model.predict_proba(self.X_test)

            confidences = [
                proba[i][pred_idx]
                for i in indices
            ]

            score_type = "probability"

        elif hasattr(clf, "decision_function"):

            scores = self.model.decision_function(self.X_test)

            # multiclass SVM
            if len(scores.shape) > 1:

                confidences = [
                    scores[i][pred_idx]
                    for i in indices
                ]

            # binary SVM
            else:

                confidences = [
                    scores[i]
                    for i in indices
                ]

            score_type = "decision_score"

        else:

            print(
                "\n⚠️ classifier does not support "
                "confidence scores"
            )

            return

        # ==========================================================
        # 2. Сортируем ошибки по уверенности
        # ==========================================================

        sorted_idx = np.argsort(confidences)[::-1][:top_n]

        # ==========================================================
        # 3. Выводим самые уверенные ошибки
        # ==========================================================

        for rank, idx_local in enumerate(sorted_idx, 1):

            global_idx = indices[idx_local]

            text = self.X_test[global_idx]
            conf = confidences[idx_local]

            print(f"\n#{rank}")
            print(f"Score type: {score_type}")
            print(f"Confidence: {conf:.4f}")

            print("\nTEXT:")
            print(text[:300])

            print(f"\nTRUE: {true_class}")
            print(f"PRED: {pred_class}")

        print("\n" + "=" * 80)
    
    # ═════════════════════════════════════════════════════════════════
    # 2. ПОИСК ПАТТЕРНОВ В ОШИБКАХ
    # ═════════════════════════════════════════════════════════════════
    
    def find_patterns(self):
        """Анализ паттернов: короткие ли отзывы? много ли отрицаний?"""
        
        print("\n" + "=" * 80)
        print("🔍 АНАЛИЗ ПАТТЕРНОВ В ОШИБКАХ")
        print("=" * 80)
        
        # Правильные vs неправильные
        correct_texts = [self.X_test[i] for i in range(len(self.X_test)) if not self.errors_mask[i]]
        error_texts = self.X_errors

        if len(error_texts) == 0:
            print("✅ Ошибок нет")
            return
        
        # 1. ДЛИНА ТЕКСТА
        print("\n1️⃣  ДЛИНА ТЕКСТА (количество слов)")
        correct_len = [len(t.split()) for t in correct_texts]
        error_len = [len(t.split()) for t in error_texts]
        
        print(f"   ✅ Правильно классифицированные:")
        print(f"      Среднее: {np.mean(correct_len):.1f} слов")
        print(f"      Медиана: {np.median(correct_len):.1f} слов")
        print(f"      Мин-Макс: {min(correct_len)}-{max(correct_len)} слов")
        
        print(f"\n   ❌ Неправильно классифицированные:")
        print(f"      Среднее: {np.mean(error_len):.1f} слов")
        print(f"      Медиана: {np.median(error_len):.1f} слов")
        print(f"      Мин-Макс: {min(error_len)}-{max(error_len)} слов")
        
        if np.mean(error_len) < np.mean(correct_len) * 0.7:
            print(f"\n   ⚠️  ПАТТЕРН: Ошибки значительно КОРОЧЕ правильных примеров!")
            print(f"       Возможно: модель не может работать с очень короткими текстами")
        
        # 2. НАЛИЧИЕ ОТРИЦАНИЙ
        print("\n2️⃣  ОТРИЦАНИЯ (не, нет, ни, никогда...)")
        negation_pattern = r'\b(не|нет|ни|никогда|ничего|никто)\b'
        
        correct_negations = sum(1 for t in correct_texts if re.search(negation_pattern, t))
        error_negations = sum(1 for t in error_texts if re.search(negation_pattern, t))
        
        print(f"   ✅ Правильно классифицированные: {correct_negations/len(correct_texts)*100:.1f}% с отрицаниями")
        print(f"   ❌ Неправильно классифицированные: {error_negations/len(error_texts)*100:.1f}% с отрицаниями")
        
        if error_negations/len(error_texts) > correct_negations/len(correct_texts) * 1.5:
            print(f"\n   ⚠️  ПАТТЕРН: Много ошибок ТАМ ГДЕ ЕСТЬ ОТРИЦАНИЯ!")
            print(f"       Возможно: обработка отрицаний не совершенна")
        
        # 3. СПЕЦИАЛЬНЫЕ СИМВОЛЫ
        print("\n3️⃣  СПЕЦИАЛЬНЫЕ СИМВОЛЫ (!!!???...)")
        special_pattern = r'[!?]{2,}|\.{2,}'
        
        correct_special = sum(1 for t in correct_texts if re.search(special_pattern, t))
        error_special = sum(1 for t in error_texts if re.search(special_pattern, t))
        
        print(f"   ✅ Правильно классифицированные: {correct_special/len(correct_texts)*100:.1f}% со спец.символами")
        print(f"   ❌ Неправильно классифицированные: {error_special/len(error_texts)*100:.1f}% со спец.символами")
        
        if error_special/len(error_texts) > correct_special/len(correct_texts) * 1.5:
            print(f"\n   ⚠️  ПАТТЕРН: Много ошибок ГДЕ ЕСТЬ СПЕЦ.СИМВОЛЫ!")
            print(f"       Возможно: лишние спец.символы путают модель")
        
        # 4. ГЛАСНЫЕ БУКВЫ (ОООЧЕНЬ)
        print("\n4️⃣  ДУБЛИРОВАНИЕ ГЛАСНЫХ (ОООЧЕНЬ, УЖАССССНО)")
        vowel_dup_pattern = r'([аеёиоуы])\1{2,}'
        
        correct_vowel = sum(1 for t in correct_texts if re.search(vowel_dup_pattern, t))
        error_vowel = sum(1 for t in error_texts if re.search(vowel_dup_pattern, t))
        
        print(f"   ✅ Правильно классифицированные: {correct_vowel/len(correct_texts)*100:.1f}% с дублями гласных")
        print(f"   ❌ Неправильно классифицированные: {error_vowel/len(error_texts)*100:.1f}% с дублями гласных")
        
        if error_vowel/len(error_texts) > correct_vowel/len(correct_texts) * 2.0:
            print(f"\n   ⚠️  ПАТТЕРН: Много ошибок С ДУБЛИРОВАНИЕМ ГЛАСНЫХ!")
            print(f"       Возможно: нужна нормализация дублей")

        print("\n" + "=" * 80)
    
    # ═════════════════════════════════════════════════════════════════
    # 3. АНАЛИЗ КОНФЛИКТУЮЩИХ ПРИЗНАКОВ
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_conflicting_features(self, top_features=15, examples_per_pair=3):
        """
        Анализ важных признаков для каждой пары ошибок
        """
        
        print("\n" + "=" * 80)
        print("🎯 АНАЛИЗ КОНФЛИКТУЮЩИХ ПРИЗНАКОВ")
        print("=" * 80)
        
        # ==========================================================
        # Получаем FeatureUnion
        # ==========================================================

        if "features" not in self.model.named_steps:
            print("\n⚠️  FeatureUnion не найден в pipeline")
            return

        features = self.model.named_steps["features"]
        clf = self.model.named_steps["clf"]

        # ==========================================================
        # Получаем word tfidf
        # ==========================================================

        word_vectorizer = None

        for name, transformer in features.transformer_list:

            if (
                isinstance(transformer, TfidfVectorizer)
                and transformer.analyzer == "word"
            ):
                word_vectorizer = transformer
                break

        if word_vectorizer is None:
            print("\n⚠️ Word TfidfVectorizer не найден")
            return

        feature_names = np.array(
            word_vectorizer.get_feature_names_out()
        )
        # Ограничиваем коэффициенты только word-feature частью
        word_feature_count = len(feature_names)
                
        # Проверяем наличие coef_ (для линейных моделей)
        if not hasattr(clf, "coef_"):
            print("\n⚠️  Classifier не имеет coef_ (нужна линейная модель)")
            return
        
        for (true_class, pred_class), indices in self.error_pairs.items():
            
            if len(indices) == 0:
                continue
            
            print(f"\n{'─' * 60}")
            print(f"🔴 TRUE CLASS = {true_class}")
            
            # Индексы классов
            classes = list(clf.classes_)
            true_idx = classes.index(true_class)
            pred_idx = classes.index(pred_class)
            
            # Коэффициенты
            if len(clf.coef_.shape) == 1:
                # Binary classification
                true_coef = clf.coef_
                pred_coef = -clf.coef_
            else:
                # Multiclass
                true_coef = clf.coef_[true_idx][:word_feature_count]
                pred_coef = clf.coef_[pred_idx][:word_feature_count]
            
            # Top признаки для true class
            true_top = np.argsort(true_coef)[-top_features:]
            
            for idx in reversed(true_top):
                print(
                    f"   {feature_names[idx]} "
                    f"({true_coef[idx]:.3f})"
                )

            print(f"\n🟢 PRED CLASS = {pred_class}")

            # Top признаки для pred class
            pred_top = np.argsort(pred_coef)[-top_features:]
            
            for idx in reversed(pred_top):
                print(
                    f"   {feature_names[idx]} "
                    f"({pred_coef[idx]:.3f})"
                )

            # Примеры ошибок
            print("\n📄 EXAMPLES:")

            for i, sample_idx in enumerate(
                indices[:examples_per_pair],
                1
            ):

                text = self.X_test[sample_idx]

                tokens = set(
                    re.findall(
                        r"\w+",
                        text.lower()
                    )
                )

                matched_true = [
                    feature_names[idx]
                    for idx in true_top
                    if feature_names[idx] in tokens
                ]

                matched_pred = [
                    feature_names[idx]
                    for idx in pred_top
                    if feature_names[idx] in tokens
                ]

                print("\n" + "-" * 60)
                print(f"Example #{i}")

                print("\nTEXT:")
                print(text[:300])

                print("\n🔴 TRUE SIGNALS FOUND:")
                print(matched_true if matched_true else "None")

                print("\n🟢 PRED SIGNALS FOUND:")
                print(matched_pred if matched_pred else "None")

        print("\n" + "=" * 80)
    
    # ═════════════════════════════════════════════════════════════════
    # 4. ВИЗУАЛИЗАЦИЯ
    # ═════════════════════════════════════════════════════════════════
    
    def visualize_errors(self):

        print("\n" + "=" * 80)
        print("📊 ВИЗУАЛИЗАЦИЯ ОШИБОК")
        print("=" * 80)

        total = len(self.y_test)
        correct = np.sum(~self.errors_mask)

        print(f"\n✅ Правильно: {correct}")
        print(
            self._make_bar(correct, total, 50)
        )

        print("\n📉 CONFUSION DISTRIBUTION:")

        for (true_class, pred_class), indices in self.error_pairs.items():

            if len(indices) == 0:
                continue

            print(
                f"\n{true_class} -> {pred_class}: "
                f"{len(indices)}"
            )

            print(
                self._make_bar(
                    len(indices),
                    total,
                    40
                )
            )

        print("\n🎯 ACCURACY BY CLASS:")

        for class_name in np.unique(self.y_test):

            total_class = np.sum(self.y_test == class_name)

            correct_class = np.sum(
                (self.y_test == class_name) &
                (self.y_pred == class_name)
            )

            acc = correct_class / total_class if total_class > 0 else 0

            print(
                f"   {class_name}: "
                f"{acc:.1%} "
                f"({correct_class}/{total_class})"
            )

    @staticmethod
    def _make_bar(value: int, total: int, width: int = 50) -> str:
        """Создаёт ASCII bar для визуализации"""
        percent = value / total if total > 0 else 0
        filled = int(width * percent)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {percent:.1%}"
    
    # ═════════════════════════════════════════════════════════════════
    # 5. ПОЛНЫЙ ОТЧЁТ
    # ═════════════════════════════════════════════════════════════════
    
    def generate_full_report(self):
        """Генерирует полный отчёт по всем анализам"""
        self.print_summary()
        self.visualize_errors()
        self.find_patterns()
        
        for (true_class, pred_class), indices in self.error_pairs.items():
            if len(indices) > 0:
                self.analyze_confusion_pair(
                    true_class,
                    pred_class,
                    top_n=5
                )
        
        self.analyze_conflicting_features(top_features=15)
        
        print("\n" + "=" * 80)
        print("✅ ОТЧЁТ ЗАВЕРШЁН")
        print("=" * 80)


# ═════════════════════════════════════════════════════════════════════
# ИНТЕГРАЦИЯ В train_model.py
# ═════════════════════════════════════════════════════════════════════

def analyze_model_errors(model, X_test, y_test, y_pred):
    """
    Быстрый анализ ошибок модели
    
    Args:
        model: sklearn pipeline
        X_test: тестовые данные
        y_test: истинные метки
        y_pred: предсказанные метки
    
    Returns:
        ErrorAnalyzer объект
    """
    analyzer = ErrorAnalyzer(model, X_test, y_test, y_pred)
    analyzer.generate_full_report()
    return analyzer


# ═════════════════════════════════════════════════════════════════════
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import joblib
    from sklearn.model_selection import train_test_split

    # Примечание: нужно импортировать из train_model.py
    # from train_model import load_training_data, preprocess_corpus

    print("=" * 80)
    print("🔎 ЗАГРУЗКА МОДЕЛИ")
    print("=" * 80)

    # Загрузка модели
    try:
        model = joblib.load("models/final_pipeline.pkl")
        print("✅ Модель загружена")
    except FileNotFoundError:
        print("❌ Модель не найдена в models/final_pipeline.pkl")
        print("Пожалуйста, обучите модель предварительно")
        exit(1)

    # Загрузка данных
    print("\n📥 Загрузка датасета...")
    
    try:
        from train_model import load_training_data, preprocess_corpus
        
        df = load_training_data()

        # Препроцессинг
        print("\n🔧 Препроцессинг текстов...")

        processed = preprocess_corpus(
            df["text"].tolist(),
            n_jobs=2
        )

        X, y = [], []

        for text, label in zip(processed, df["sentiment"]):
            if text.strip():
                X.append(text)
                y.append(label)

        # Воссоздаём test split
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        print(f"\n📊 Test examples: {len(X_test)}")

        # Predict
        print("\n🤖 Предсказания модели...")
        y_pred = model.predict(X_test)

        # Анализ ошибок
        print("\n🔍 Запуск анализа ошибок...")

        analyzer = ErrorAnalyzer(
            model,
            X_test,
            y_test,
            y_pred
        )

        analyzer.generate_full_report()
        
    except ImportError:
        print("❌ Не удается импортировать train_model")
        print("Пожалуйста, убедитесь что train_model.py доступен")