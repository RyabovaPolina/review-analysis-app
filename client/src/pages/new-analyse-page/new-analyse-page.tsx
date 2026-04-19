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

export default function NewAnalysePage() {
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

  function handleSubmit() {
    if (!file) return;
    dispatch(uploadFilePreview(file));
  }

  useEffect(() => {
    if (headers && preview) {
      navigate('/upload-process')
    }
  }, [headers, preview]);

  // 👇 Автоматический запуск анализа после подтверждения маппинга
  useEffect(() => {
    if (mappingConfirmed && fileId && !analyzing && !analysisResults) {
      console.log("Запуск анализа для fileId:", fileId);
      dispatch(startAnalysis(fileId));
    }
  }, [mappingConfirmed, fileId, analyzing, analysisResults, dispatch]);

  async function handleNext(mapping: Record<RequiredFieldKey, string>) {
    if (!fileId) return;

    try {
      await dispatch(confirmColumnMapping({ fileId, mapping })).unwrap();
      setShowPopup(false);
      // Анализ запустится автоматически через useEffect выше
    } catch (err) {
      console.error("Ошибка подтверждения маппинга:", err);
    }
  }

  function handleClosePopup() {
    setShowPopup(false);
    dispatch(clearAnalyseState());
    setFile(null);
  }

  function handleReset() {
    dispatch(clearAnalyseState());
    setFile(null);
  }

  return (
    <>
      <div className="container-page-content new-analyse">
        <div className="container-header">
          <div className="step-container main">
            <div className="step">1</div>
            <h2>Загрузка файла</h2>
          </div>
          <div className="step-container">
            <div className="step">2</div>
            <h2>Обработка файла</h2>
          </div>
          <div className="step-container">
            <div className="step">3</div>
            <h2>Результат</h2>
          </div>
        </div>
        {/* Форма загрузки */}
        {!mappingConfirmed && !analyzing && !analysisResults && (
          <div className="auth-form">
            <input
              type="file"
              id="csv-file"
              name="csv"
              accept=".csv"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  setFile(e.target.files[0]);
                }
              }}
            />
            <button onClick={handleSubmit} disabled={!file || loading}>
              {loading ? "Загрузка..." : "Загрузить csv"}
            </button>
            {error && <p className="error-message">{error}</p>}
          </div>
        )}
      </div>
    </>
  );
}
