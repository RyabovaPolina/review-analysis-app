import { useState } from "react";
import { useAppDispatch, useAppSelector } from "../../hooks";
import { clearAnalyseState } from "../../store/slices/analyse-slice";
import "./style.css";
import type { RootState } from "../../store";
import { useNavigate } from "react-router-dom";

const safeFixed = (value?: number, digits = 2) => {
  return typeof value === "number" && !isNaN(value)
    ? value.toFixed(digits)
    : "—";
};

const percent = (value?: number, digits = 1) => {
  return typeof value === "number" && !isNaN(value)
    ? `${value.toFixed(digits)}%`
    : "—";
};

export default function ResultPage() {
  const [file, setFile] = useState<File | null>(null);
  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  const { analyzing, analysisResults } = useAppSelector(
    (state: RootState) => state.analyse
  );

  function handleReset() {
    dispatch(clearAnalyseState());
    setFile(null);
    navigate("/new-analyse");
  }

  if (analysisResults) {
    console.log("analysis:", analysisResults);
  }

  const summary = analysisResults?.summary;
  const sentiment = analysisResults?.sentiment_analysis;
  const keywords = analysisResults?.keywords;

  return (
    <div className="container-page-content">
      <div className="container-header">
        <div className="step-container">
          <div className="step">1</div>
          <h2>Загрузка файла</h2>
        </div>
        <div className="step-container">
          <div className="step">2</div>
          <h2>Обработка файла</h2>
        </div>
        <div className="step-container main">
          <div className="step">3</div>
          <h2>Результат</h2>
        </div>
      </div>

      {/* Индикатор анализа */}
      {analyzing && (
        <div className="analysis-progress">
          <div className="spinner"></div>
          <p>Анализируем отзывы... Это может занять некоторое время.</p>
        </div>
      )}

      {/* Результаты */}
      {analysisResults && summary && sentiment && (
        <div className="analysis-results">
          <div className="results-grid">
            <div className="result-card positive">
              <h3>Позитивные</h3>
              <p className="result-number">
                {summary?.positive_count ?? "—"}
              </p>
              <p className="result-percentage">
                {percent(summary?.positive_pct)}
              </p>
            </div>

            <div className="result-card negative">
              <h3>Негативные</h3>
              <p className="result-number">
                {summary?.negative_count ?? "—"}
              </p>
              <p className="result-percentage">
                {percent(summary?.negative_pct)}
              </p>
            </div>

            <div className="result-card neutral">
              <h3>Нейтральные</h3>
              <p className="result-number">
                {summary?.neutral_count ?? "—"}
              </p>
              <p className="result-percentage">
                {percent(summary?.neutral_pct)}
              </p>
            </div>
          </div>

          <div className="result-summary">
            <p>
              Всего отзывов:{" "}
              <strong>{summary?.total_reviews ?? "—"}</strong>
            </p>
            <p>
              Средняя оценка тональности:{" "}
              <strong>
                {safeFixed(sentiment?.avg_sentiment_score, 2)}
              </strong>
            </p>
          </div>

          {/* KEYWORDS */}
          {keywords && (
            <div className="keywords-section">
              <h2>Ключевые слова по тональности</h2>

              {/* POSITIVE */}
              <div className="keywords-group positive">
                <h3>Позитивные отзывы</h3>

                <div className="keywords-subgroup">
                  <h4>Частые слова</h4>
                  <ul>
                    {keywords?.positive?.unigrams?.map((item) => (
                      <li key={item.text}>
                        {item.text} ({safeFixed(item.score, 2)})
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="keywords-subgroup">
                  <h4>Частые словосочетания</h4>
                  <ul>
                    {keywords?.positive?.bigrams?.map((item) => (
                      <li key={item.text}>
                        {item.text} ({safeFixed(item.score, 2)})
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* NEGATIVE */}
              <div className="keywords-group negative">
                <h3>Негативные отзывы</h3>

                <div className="keywords-subgroup">
                  <h4>Частые слова</h4>
                  <ul>
                    {keywords?.negative?.unigrams?.map((item) => (
                      <li key={item.text}>
                        {item.text} ({safeFixed(item.score, 2)})
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="keywords-subgroup">
                  <h4>Частые словосочетания</h4>
                  <ul>
                    {keywords?.negative?.bigrams?.map((item) => (
                      <li key={item.text}>
                        {item.text} ({safeFixed(item.score, 2)})
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* NEUTRAL */}
              <div className="keywords-group neutral">
                <h3>Нейтральные отзывы</h3>

                <div className="keywords-subgroup">
                  <h4>Частые слова</h4>
                  <ul>
                    {keywords?.neutral?.unigrams?.map((item) => (
                      <li key={item.text}>
                        {item.text} ({safeFixed(item.score, 2)})
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="keywords-subgroup">
                  <h4>Частые словосочетания</h4>
                  <ul>
                    {keywords?.neutral?.bigrams?.map((item) => (
                      <li key={item.text}>
                        {item.text} ({safeFixed(item.score, 2)})
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* 🔴 ТОП ПРОБЛЕМЫ */}
          {analysisResults?.top_problems?.length > 0 && (
            <div className="problems-section">
              <h2>Главные проблемы</h2>
              <div className="problems-list">
                {analysisResults.top_problems.map((item, index) => (
                  <div key={index} className="problem-card">
                    <div className="problem-header">
                      <strong>{index + 1}. {item.aspect}</strong>
                      <span className={`severity ${item.severity}`}>
                        {item.severity}
                      </span>
                    </div>

                    <div className="problem-stats">
                      <p>Доля: {percent(item.coverage_pct)}</p>
                      <p>Упоминаний: {item.mentions}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 🎯 РЕКОМЕНДАЦИИ */}
          {analysisResults?.recommendations?.length > 0 && (
            <div className="recommendations-section">
              <h2>Рекомендации</h2>
              <div className="recommendations-list">
                {analysisResults.recommendations.map((rec, index) => (
                  <div key={index} className={`recommendation-card ${rec.type}`}>
                    <div className="recommendation-header">
                      <strong>{rec.aspect}</strong>
                      <span className={`priority ${rec.priority}`}>
                        {rec.priority}
                      </span>
                    </div>

                    {/* проблема или сильная сторона */}
                    {rec.issue && <p className="rec-issue">{rec.issue}</p>}
                    {rec.strength && <p className="rec-strength">{rec.strength}</p>}

                    {/* действия */}
                    <pre className="rec-action">
                      {rec.action}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          )}

          <button onClick={handleReset} className="btn-secondary">
            Загрузить новый файл
          </button>
        </div>
      )}
    </div>
  );
}