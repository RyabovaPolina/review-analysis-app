import { useEffect, useState } from "react";
import { useAppDispatch, useAppSelector } from "../../hooks";
import {
  uploadFilePreview,
  confirmColumnMapping,
  clearAnalyseState,
  startAnalysis,
} from "../../store/slices/analyse-slice";
import "./style.css";
import type { RootState } from "../../store";
import Popup, { type RequiredFieldKey } from "../../components/popup/popup";
import { useNavigate } from "react-router-dom";

export default function ResultPage() {
  const [file, setFile] = useState<File | null>(null);
  const dispatch = useAppDispatch();
  const [showPopup, setShowPopup] = useState(false);
  const navigate = useNavigate();

  const {
    fileId,
    headers,
    preview,
    loading,
    mappingConfirmed,
    analyzing,
    analysisResults,
    error,
  } = useAppSelector((state: RootState) => state.analyse);

  function handleReset() {
    dispatch(clearAnalyseState());
    setFile(null);
    navigate('/new-analyse')
  }

  return (
    <>
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

        {/* Результаты анализа */}
        {analysisResults && (
          <div className="analysis-results">
            <div className="results-grid">
              <div className="result-card positive">
                <h3>Позитивные</h3>
                <p className="result-number">
                  {analysisResults.positive_count}
                </p>
                <p className="result-percentage">
                  {(
                    (analysisResults.positive_count /
                      analysisResults.total_reviews) *
                    100
                  ).toFixed(1)}
                  %
                </p>
              </div>
              <div className="result-card negative">
                <h3>Негативные</h3>
                <p className="result-number">
                  {analysisResults.negative_count}
                </p>
                <p className="result-percentage">
                  {(
                    (analysisResults.negative_count /
                      analysisResults.total_reviews) *
                    100
                  ).toFixed(1)}
                  %
                </p>
              </div>
              <div className="result-card neutral">
                <h3>Нейтральные</h3>
                <p className="result-number">{analysisResults.neutral_count}</p>
                <p className="result-percentage">
                  {(
                    (analysisResults.neutral_count /
                      analysisResults.total_reviews) *
                    100
                  ).toFixed(1)}
                  %
                </p>
              </div>
            </div>
            <div className="result-summary">
              <p>
                Всего отзывов: <strong>{analysisResults.total_reviews}</strong>
              </p>
              <p>
                Средняя оценка тональности:{" "}
                <strong>
                  {analysisResults.avg_sentiment_score.toFixed(2)}
                </strong>
              </p>
            </div>
            <button onClick={handleReset} className="btn-secondary">
              Загрузить новый файл
            </button>
          </div>
        )}
      </div>
    </>
  );
}
