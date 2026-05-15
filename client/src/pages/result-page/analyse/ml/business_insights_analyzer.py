"""
✅ INTEGRATED ASPECT ANALYSIS WITH BUSINESS INSIGHTS
Полный анализ аспектов с рекомендациями для владельцев бизнеса

ИСПРАВЛЕНИЕ: Добавлена поддержка разных имён колонок с текстами
"""

from ml.aspect_extractor import AspectExtractor, aggregate_aspects_by_sentiment
from collections import defaultdict
import json
from typing import Dict, List, Tuple
import numpy as np


class BusinessInsightAnalyzer:
    """
    Анализирует отзывы и выводит рекомендации для владельцев бизнеса.
    """
    
    def __init__(self):
        self.aspect_extractor = AspectExtractor()
    
    def analyze_reviews_by_aspects(
        self,
        df_with_sentiment,  # DataFrame с колонками: текстов и 'sentiment'
        text_column="processed"  # ✅ ИСПРАВЛЕНИЕ: параметр для гибкости
    ) -> Dict:
        """
        Полный анализ отзывов по аспектам.
        
        Args:
            df_with_sentiment: DataFrame с колонками текстов и 'sentiment'
            text_column: название колонки с текстами (по умолчанию 'processed')
        
        Returns:
            {
                "summary": {...},
                "aspect_analysis": {...},
                "top_problems": [...],
                "top_strengths": [...],
                "recommendations": [...],
            }
        """
        
        # ─────────────────────────────────────────────────────────
        # ПОДГОТОВКА ДАННЫХ
        # ─────────────────────────────────────────────────────────
        
        # ✅ ИСПРАВЛЕНИЕ: Проверяем какая колонка доступна
        if text_column not in df_with_sentiment.columns:
            # Пробуем альтернативные имена
            if "processed_text" in df_with_sentiment.columns:
                text_column = "processed_text"
            elif "processed" in df_with_sentiment.columns:
                text_column = "processed"
            else:
                raise ValueError(
                    f"Не найдена колонка с текстами '{text_column}'. "
                    f"Доступные: {df_with_sentiment.columns.tolist()}"
                )
        
        texts_by_sentiment = {
            "positive": df_with_sentiment[df_with_sentiment["sentiment"] == "positive"][text_column].tolist(),
            "negative": df_with_sentiment[df_with_sentiment["sentiment"] == "negative"][text_column].tolist(),
            "neutral": df_with_sentiment[df_with_sentiment["sentiment"] == "neutral"][text_column].tolist(),
        }
        
        # ─────────────────────────────────────────────────────────
        # АГРЕГАЦИЯ АСПЕКТОВ
        # ─────────────────────────────────────────────────────────
        
        aggregated = aggregate_aspects_by_sentiment(texts_by_sentiment, self.aspect_extractor)
        
        # ─────────────────────────────────────────────────────────
        # ВЫЯВЛЕНИЕ ПРОБЛЕМНЫХ МЕСТ (TOP PROBLEMS)
        # ─────────────────────────────────────────────────────────
        
        problems = self._identify_problems(aggregated)
        
        # ─────────────────────────────────────────────────────────
        # ВЫЯВЛЕНИЕ СИЛЬНЫХ СТОРОН (TOP STRENGTHS)
        # ─────────────────────────────────────────────────────────
        
        strengths = self._identify_strengths(aggregated)
        
        # ─────────────────────────────────────────────────────────
        # ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ
        # ─────────────────────────────────────────────────────────
        
        recommendations = self._generate_recommendations(problems, strengths)
        
        # ─────────────────────────────────────────────────────────
        # ФИНАЛЬНЫЙ ОТЧЕТ
        # ─────────────────────────────────────────────────────────
        
        return {
            "summary": {
                "total_reviews": len(df_with_sentiment),
                "positive_count": len(texts_by_sentiment["positive"]),
                "negative_count": len(texts_by_sentiment["negative"]),
                "neutral_count": len(texts_by_sentiment["neutral"]),
                "positive_pct": len(texts_by_sentiment["positive"]) / len(df_with_sentiment) * 100,
                "negative_pct": len(texts_by_sentiment["negative"]) / len(df_with_sentiment) * 100,
                "neutral_pct": len(texts_by_sentiment["neutral"]) / len(df_with_sentiment) * 100,
            },
            "aspect_analysis": aggregated,
            "top_problems": problems,
            "top_strengths": strengths,
            "recommendations": recommendations,
        }
    
    def _identify_problems(self, aggregated: Dict) -> List[Dict]:
        """
        Выявляет проблемные аспекты (много негативных упоминаний).
        """
        problems = []
        
        negative_aspects = aggregated.get("negative", {})
        
        for aspect, stats in negative_aspects.items():
            mentions = stats["mentions"]
            coverage = stats["coverage"]
            
            # Считаем проблемность: частота + покрытие (процент затронутых людей)
            issue_score = min(1.0, (mentions / 1000 + coverage) / 2)
            
            # Определяем серьезность
            if issue_score > 0.7:
                severity = "critical"
            elif issue_score > 0.4:
                severity = "high"
            else:
                severity = "medium"
            
            problems.append({
                "aspect": aspect,
                "issue_score": round(issue_score, 3),
                "mentions": mentions,
                "coverage_pct": round(coverage * 100, 1),
                "severity": severity,
                "avg_sentiment": round(stats["avg_sentiment"], 3),
            })
        
        # Сортируем по серьезности
        problems.sort(key=lambda x: (
            {"critical": 0, "high": 1, "medium": 2}[x["severity"]],
            -x["mentions"]
        ))
        
        return problems[:5]  # Top 5 проблем
    
    def _identify_strengths(self, aggregated: Dict) -> List[Dict]:
        """
        Выявляет сильные стороны (много позитивных упоминаний).
        """
        strengths = []
        
        positive_aspects = aggregated.get("positive", {})
        
        for aspect, stats in positive_aspects.items():
            mentions = stats["mentions"]
            coverage = stats["coverage"]
            avg_sentiment = stats["avg_sentiment"]
            
            # Считаем силу: частота + покрытие + интенсивность тональности
            strength_score = min(1.0, (
                (mentions / 1000) * 0.5 +  # Частота
                (coverage) * 0.3 +  # Покрытие
                (max(0, avg_sentiment)) * 0.2  # Интенсивность
            ))
            
            strengths.append({
                "aspect": aspect,
                "strength_score": round(strength_score, 3),
                "mentions": mentions,
                "coverage_pct": round(coverage * 100, 1),
                "avg_sentiment": round(avg_sentiment, 3),
            })
        
        strengths.sort(key=lambda x: -x["mentions"])
        
        return strengths[:5]  # Top 5 сильных сторон
    
    def _generate_recommendations(
        self,
        problems: List[Dict],
        strengths: List[Dict]
    ) -> List[Dict]:
        """
        Генерирует конкретные рекомендации на основе проблем и сильных сторон.
        """
        
        recommendations_by_aspect = {
            "еда": {
                "problem": "📍 Улучшить качество/вкус еды:\n"
                           "  • Пересмотреть рецепты\n"
                           "  • Проверить свежесть ингредиентов\n"
                           "  • Провести дегустацию с фокус-группой\n"
                           "  • Нанять опытного шефа",
                "strength": "✨ Ваша еда - основной конкурентный преимущество!\n"
                           "  • Поддерживайте качество\n"
                           "  • Продвигайте это в рекламе\n"
                           "  • Создавайте специальные блюда"
            },
            "персонал": {
                "problem": "📍 Улучшить обслуживание:\n"
                           "  • Обучение персонала вежливости\n"
                           "  • Усиление контроля качества\n"
                           "  • Работа с мотивацией сотрудников\n"
                           "  • Mystery shopping программа",
                "strength": "✨ Ваш персонал отличный!\n"
                           "  • Выделяйте бонусы за отзывы\n"
                           "  • Рассказывайте о команде в маркетинге\n"
                           "  • Удерживайте таланты"
            },
            "цена": {
                "problem": "📍 Работать с ценовой позицией:\n"
                           "  • Анализ конкурентов\n"
                           "  • Ревью себестоимости\n"
                           "  • Введение скидок/акций\n"
                           "  • Бизнес-комбо меню",
                "strength": "✨ Ваша цена конкурентна!\n"
                           "  • Подчеркивайте в маркетинге\n"
                           "  • Программа лояльности\n"
                           "  • Ценовые гарантии"
            },
            "атмосфера": {
                "problem": "📍 Улучшить атмосферу:\n"
                           "  • Ремонт интерьера\n"
                           "  • Уборка и уход за чистотой\n"
                           "  • Музыка и освещение\n"
                           "  • Профессиональный дизайнер",
                "strength": "✨ Ваша атмосфера привлекает людей!\n"
                           "  • Инвестируйте в поддержание\n"
                           "  • Фотографируйте для соцсетей\n"
                           "  • Проводите ивенты"
            },
            "расположение": {
                "problem": "📍 Работать с доступностью:\n"
                           "  • Рассказывать как добраться\n"
                           "  • Парковка или велопарковка\n"
                           "  • Улучшить видимость вывески\n"
                           "  • Работать с маршрутами доставки",
                "strength": "✨ Отличное место расположения!\n"
                           "  • Максимизируйте пешеходный трафик\n"
                           "  • Уличная реклама\n"
                           "  • Партнерства с близлежащими"
            },
            "доставка": {
                "problem": "📍 Ускорить доставку/приготовление:\n"
                           "  • Оптимизация процессов\n"
                           "  • Дополнительный персонал\n"
                           "  • Система электронных заказов\n"
                           "  • Аналитика времени приготовления",
                "strength": "✨ Быстрая доставка - ваш плюс!\n"
                           "  • Гарантии по времени доставки\n"
                           "  • Express сервис премиум\n"
                           "  • Говорите об этом везде"
            },
        }
        
        result_recommendations = []
        
        # Добавляем рекомендации по проблемам
        for problem in problems[:3]:  # Top 3
            aspect = problem["aspect"]
            if aspect in recommendations_by_aspect:
                result_recommendations.append({
                    "type": "problem",
                    "aspect": aspect,
                    "priority": "high" if problem["severity"] == "critical" else "medium",
                    "issue": f"{aspect.upper()}: {problem['coverage_pct']:.1f}% отзывов содержат критику",
                    "action": recommendations_by_aspect[aspect]["problem"]
                })
        
        # Добавляем рекомендации по сильным сторонам
        for strength in strengths[:2]:  # Top 2
            aspect = strength["aspect"]
            if aspect in recommendations_by_aspect:
                result_recommendations.append({
                    "type": "opportunity",
                    "aspect": aspect,
                    "priority": "medium",
                    "strength": f"{aspect.upper()}: {strength['coverage_pct']:.1f}% отзывов хвалят",
                    "action": recommendations_by_aspect[aspect]["strength"]
                })
        
        return result_recommendations


# ═════════════════════════════════════════════════════════════════════
# ФОРМАТИРОВАНИЕ ДЛЯ ВЫВОДА
# ═════════════════════════════════════════════════════════════════════

def format_analysis_for_display(analysis: Dict) -> str:
    """
    Форматирует анализ для красивого вывода.
    """
    
    output = []
    
    # ─────────────────────────────────────────────────────────
    # ОСНОВНАЯ СТАТИСТИКА
    # ─────────────────────────────────────────────────────────
    
    summary = analysis["summary"]
    output.append("=" * 100)
    output.append("📊 ОСНОВНАЯ СТАТИСТИКА")
    output.append("=" * 100)
    output.append(f"\nВсего отзывов: {summary['total_reviews']:,}")
    output.append(f"  ✅ Позитивные: {summary['positive_count']:,} ({summary['positive_pct']:.1f}%)")
    output.append(f"  ❌ Негативные: {summary['negative_count']:,} ({summary['negative_pct']:.1f}%)")
    output.append(f"  😐 Нейтральные: {summary['neutral_count']:,} ({summary['neutral_pct']:.1f}%)")
    
    # ─────────────────────────────────────────────────────────
    # ПРОБЛЕМНЫЕ МЕСТА
    # ─────────────────────────────────────────────────────────
    
    output.append("\n" + "=" * 100)
    output.append("🚨 ПРОБЛЕМНЫЕ МЕСТА (ТОП-5)")
    output.append("=" * 100)
    
    for i, problem in enumerate(analysis["top_problems"], 1):
        severity_icon = "🔴" if problem["severity"] == "critical" else "🟠"
        output.append(f"\n{i}. {severity_icon} {problem['aspect'].upper()}")
        output.append(f"   Упоминаний: {problem['mentions']:,} ({problem['coverage_pct']:.1f}% отзывов)")
        output.append(f"   Серьезность: {problem['severity'].upper()}")
        output.append(f"   Средняя оценка: {problem['avg_sentiment']:+.2f}")
    
    # ─────────────────────────────────────────────────────────
    # СИЛЬНЫЕ СТОРОНЫ
    # ─────────────────────────────────────────────────────────
    
    output.append("\n" + "=" * 100)
    output.append("⭐ СИЛЬНЫЕ СТОРОНЫ (ТОП-5)")
    output.append("=" * 100)
    
    for i, strength in enumerate(analysis["top_strengths"], 1):
        output.append(f"\n{i}. ✨ {strength['aspect'].upper()}")
        output.append(f"   Упоминаний: {strength['mentions']:,} ({strength['coverage_pct']:.1f}% отзывов)")
        output.append(f"   Оценка: {strength['avg_sentiment']:+.2f}")
    
    # ─────────────────────────────────────────────────────────
    # РЕКОМЕНДАЦИИ
    # ─────────────────────────────────────────────────────────
    
    output.append("\n" + "=" * 100)
    output.append("💡 РЕКОМЕНДАЦИИ")
    output.append("=" * 100)
    
    for rec in analysis["recommendations"]:
        if rec["type"] == "problem":
            output.append(f"\n🎯 {rec['issue']}")
        else:
            output.append(f"\n✨ {rec['strength']}")
        
        output.append(rec["action"])
    
    return "\n".join(output)