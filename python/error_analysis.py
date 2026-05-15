"""
ERROR ANALYSIS: Подробный анализ false positives и false negatives
- Какие тексты классифицируются неправильно?
- Есть ли паттерны в ошибках?
- Какие признаки вызывают конфликты?
- Визуализация проблемных примеров
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


# ═════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС АНАЛИЗА ОШИБОК
# ═════════════════════════════════════════════════════════════════════

class ErrorAnalyzer:
    """
    Анализ ошибок классификации с поиском паттернов и проблемных признаков
    
    Использование:
        analyzer = ErrorAnalyzer(model, X_test, y_test, y_pred)
        analyzer.print_summary()
        analyzer.analyze_false_positives()
        analyzer.analyze_false_negatives()
        analyzer.find_patterns()
        analyzer.visualize_errors()
    """
    
    def __init__(self, model: Pipeline, X_test: List[str], y_test: np.ndarray, y_pred: np.ndarray):
        """
        Args:
            model: обученный sklearn pipeline
            X_test: тестовые тексты
            y_test: истинные метки
            y_pred: предсказанные метки
        """
        self.model = model
        # 🔥 ВАЖНО: приведение к numpy
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
        """
        Анализ ошибок между двумя классами

        Например:
            negative -> positive
            neutral -> positive
            positive -> neutral
        """

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

        classes = list(self.model.classes_)
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
    # 2. АНАЛИЗ FALSE POSITIVES
    # ═════════════════════════════════════════════════════════════════
     
    # ═════════════════════════════════════════════════════════════════
    # 4. ПОИСК ПАТТЕРНОВ В ОШИБКАХ
    # ═════════════════════════════════════════════════════════════════
    
    def find_patterns(self):
        """Анализ паттернов: короткие ли отзывы? много ли отрицаний?"""
        
        print("\n" + "=" * 80)
        print("🔍 АНАЛИЗ ПАТТЕРНОВ В ОШИБКАХ")
        print("=" * 80)
        
        # Правильные vs неправильные
        correct_texts = [self.X_test[i] for i in range(len(self.X_test)) if not self.errors_mask[i]]
        error_texts = self.X_errors
        
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
        
        if len(error_texts) == 0:
            print("✅ Ошибок нет")
            return
        elif error_negations/len(error_texts) > correct_negations/len(correct_texts) * 1.5:
            print(f"\n   ⚠️  ПАТТЕРН: Много ошибок ТАМ ГДЕ ЕСТЬ ОТРИЦАНИЯ!")
            print(f"       Возможно: обработка отрицаний не совершенна")
        
        # 3. СПЕЦИАЛЬНЫЕ СИМВОЛЫ
        print("\n3️⃣  СПЕЦИАЛЬНЫЕ СИМВОЛЫ (!!!???...)")
        special_pattern = r'[!?]{2,}|\.{2,}'
        
        correct_special = sum(1 for t in correct_texts if re.search(special_pattern, t))
        error_special = sum(1 for t in error_texts if re.search(special_pattern, t))
        
        print(f"   ✅ Правильно классифицированные: {correct_special/len(correct_texts)*100:.1f}% со спец.символами")
        print(f"   ❌ Неправильно классифицированные: {error_special/len(error_texts)*100:.1f}% со спец.символами")
        
        # 4. ГЛАСНЫЕ БУКВЫ (ОООЧЕНЬ)
        print("\n4️⃣  ДУБЛИРОВАНИЕ ГЛАСНЫХ (ОООЧЕНЬ, УЖАССССНО)")
        vowel_dup_pattern = r'([аеёиоуы])\1{2,}'
        
        correct_vowel_dup = sum(1 for t in correct_texts if re.search(vowel_dup_pattern, t))
        error_vowel_dup = sum(1 for t in error_texts if re.search(vowel_dup_pattern, t))
        
        print(f"   ✅ Правильно классифицированные: {correct_vowel_dup/len(correct_texts)*100:.1f}% с дублями")
        print(f"   ❌ Неправильно классифицированные: {error_vowel_dup/len(error_texts)*100:.1f}% с дублями")
        
        # 5. СЛЕНГ
        print("\n5️⃣  МАГАЗИННЫЙ СЛЕНГ (пушка, шляпа, барахло...)")
        slang_words = {'пушка', 'бомба', 'огонь', 'шляпа', 'барахло', 'мусор', 'хлам', 
                      'конфетка', 'копейки', 'грабёж', 'молния', 'черепаха'}
        
        correct_slang = sum(1 for t in correct_texts if any(word in t.lower() for word in slang_words))
        error_slang = sum(1 for t in error_texts if any(word in t.lower() for word in slang_words))
        
        print(f"   ✅ Правильно классифицированные: {correct_slang/len(correct_texts)*100:.1f}% со сленгом")
        print(f"   ❌ Неправильно классифицированные: {error_slang/len(error_texts)*100:.1f}% со сленгом")
        
        print("\n" + "=" * 80)
    
    # ═════════════════════════════════════════════════════════════════
    # 5. АНАЛИЗ CONFLICTING ПРИЗНАКОВ
    # ═════════════════════════════════════════════════════════════════
    
    def analyze_conflicting_features(
        self,
        top_features=15,
        examples_per_pair=3
    ):
        """
        Анализ конфликтующих признаков
        для каждого confusion pair.

        Показывает:
        - какие признаки толкают в TRUE класс
        - какие признаки толкают в PREDICTED класс
        - какие признаки реально встретились в тексте
        """

        print("\n" + "=" * 80)
        print("⚔️ CONFLICTING FEATURE ANALYSIS")
        print("=" * 80)

        # ==========================================================
        # Проверка classifier
        # ==========================================================

        clf = self.model.named_steps["clf"]

        if not hasattr(clf, "coef_"):

            print("⚠️ classifier has no coef_")
            return

        # ==========================================================
        # Получаем vectorizers
        # ==========================================================

        features = self.model.named_steps["features"]

        word_vectorizer = dict(
            features.transformer_list
        )["word"]

        char_vectorizer = dict(
            features.transformer_list
        )["char"]

        # ==========================================================
        # Feature names
        # ==========================================================

        word_features = (
            word_vectorizer.get_feature_names_out()
        )

        char_features = (
            char_vectorizer.get_feature_names_out()
        )

        feature_names = np.concatenate([
            word_features,
            char_features
        ])

        # ==========================================================
        # Coefficients
        # ==========================================================

        coef = clf.coef_
        classes = list(clf.classes_)

        # Safety
        min_len = min(
            len(feature_names),
            coef.shape[1]
        )

        feature_names = feature_names[:min_len]
        coef = coef[:, :min_len]

        # ==========================================================
        # Анализируем ВСЕ confusion pairs
        # ==========================================================

        for (true_class, pred_class), indices in self.error_pairs.items():

            if len(indices) == 0:
                continue

            print("\n" + "=" * 80)
            print(
                f"⚠️ CONFLICT: "
                f"{true_class} -> {pred_class}"
            )
            print("=" * 80)

            # ------------------------------------------------------
            # Индексы классов
            # ------------------------------------------------------

            true_idx = classes.index(true_class)
            pred_idx = classes.index(pred_class)

            true_coef = coef[true_idx]
            pred_coef = coef[pred_idx]

            # ------------------------------------------------------
            # ТОП признаков классов
            # ------------------------------------------------------

            true_top = np.argsort(true_coef)[-top_features:]
            pred_top = np.argsort(pred_coef)[-top_features:]

            true_tokens = set(
                feature_names[idx]
                for idx in true_top
            )

            pred_tokens = set(
                feature_names[idx]
                for idx in pred_top
            )

            print(f"\n🔴 TRUE CLASS = {true_class}")

            for idx in reversed(true_top):

                print(
                    f"   {feature_names[idx]} "
                    f"({true_coef[idx]:.3f})"
                )

            print(f"\n🟢 PRED CLASS = {pred_class}")

            for idx in reversed(pred_top):

                print(
                    f"   {feature_names[idx]} "
                    f"({pred_coef[idx]:.3f})"
                )

            # ------------------------------------------------------
            # Анализируем тексты ошибок
            # ------------------------------------------------------

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
                    token
                    for token in true_tokens
                    if any(token in t for t in tokens)
                ]

                matched_pred = [
                    token
                    for token in pred_tokens
                    if any(token in t for t in tokens)
                ]

                print("\n" + "-" * 60)
                print(f"Example #{i}")

                print("\nTEXT:")
                print(text[:300])

                print("\n🔴 TRUE SIGNALS FOUND:")
                print(
                    matched_true[:15]
                    if matched_true else "None"
                )

                print("\n🟢 PRED SIGNALS FOUND:")
                print(
                    matched_pred[:15]
                    if matched_pred else "None"
                )

        print("\n" + "=" * 80)
    # ═════════════════════════════════════════════════════════════════
    # 6. ВИЗУАЛИЗАЦИЯ
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

            acc = correct_class / total_class

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
    # 7. ПОЛНЫЙ ОТЧЁТ
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
    Удобная функция для интеграции в train_model.py
    
    Использование в train_model.py:
    
    pred = best_model.predict(X_test)
    from error_analysis import analyze_model_errors
    analyze_model_errors(best_model, X_test, y_test, pred)
    """
    analyzer = ErrorAnalyzer(model, X_test, y_test, y_pred)
    analyzer.generate_full_report()
    return analyzer


# ═════════════════════════════════════════════════════════════════════
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Демонстрация (требует обученную модель и тестовые данные):
    
    from error_analysis import ErrorAnalyzer
    from train_model import train_final_model
    
    # Обучение модели
    model = train_final_model(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Анализ ошибок
    y_pred = model.predict(X_test)
    analyzer = ErrorAnalyzer(model, X_test, y_test, y_pred)
    analyzer.generate_full_report()
    """
    print("""
    Error Analysis Module
    ════════════════════════════════════════════════════════════════════
    
    Использование:
    
    1. В train_model.py (в конце):
    
        from error_analysis import analyze_model_errors
        
        pred = best_model.predict(X_test)
        analyze_model_errors(best_model, X_test, y_test, pred)
    
    2. Отдельно:
    
        from error_analysis import ErrorAnalyzer
        
        analyzer = ErrorAnalyzer(model, X_test, y_test, y_pred)
        analyzer.generate_full_report()
        
        # Или отдельные анализы:
        analyzer.print_summary()
        analyzer.analyze_false_positives()
        analyzer.analyze_false_negatives()
        analyzer.find_patterns()
        analyzer.analyze_conflicting_features()
    
    Методы:
    ──────────────────────────────────────────────────────────────────
    • print_summary()              - общая статистика по ошибкам
    • analyze_false_positives()    - анализ ложных положительных
    • analyze_false_negatives()    - анализ ложных отрицательных
    • find_patterns()              - поиск паттернов в ошибках
    • analyze_conflicting_features() - анализ противоречащих признаков
    • visualize_errors()           - ASCII визуализация
    • generate_full_report()       - полный отчёт
    
    ════════════════════════════════════════════════════════════════════
    """)