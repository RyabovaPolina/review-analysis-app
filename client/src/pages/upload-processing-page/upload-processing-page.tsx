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
import { useNavigate } from "react-router-dom";

type PopupData = {
  headers: string[];
  preview: Record<string, string>[];
  onClose: () => void;
  onNext: (mapping: Record<RequiredFieldKey, string>) => void;
};

const requiredFields = {
  text: "Текст отзыва",
  rating: "Рейтинг",
};

export type RequiredFieldKey = keyof typeof requiredFields;

export default function UploadProcessingPage() {
  const dispatch = useAppDispatch();
  const [mapping, setMapping] = useState<Record<RequiredFieldKey, string>>({
    text: "",
    rating: "",
  });
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

  // 👇 Автоматический запуск анализа после подтверждения маппинга
  useEffect(() => {
    if (mappingConfirmed && fileId && !analyzing && !analysisResults) {
      console.log("Запуск анализа для fileId:", fileId);
      dispatch(startAnalysis(fileId));
    }
  }, [mappingConfirmed, fileId, analyzing, analysisResults, dispatch]);

  // 👇 Редирект на /result когда анализ начался
  useEffect(() => {
    if (analyzing || analysisResults) {
      console.log("Переход на страницу результатов");
      navigate("/result");
    }
  }, [analyzing, analysisResults]);

  async function handleNext() {
    if (!fileId || !mapping.text) return;

    try {
      console.log("Подтверждение маппинга:", mapping);
      await dispatch(confirmColumnMapping({ fileId, mapping })).unwrap();
      // Анализ запустится автоматически через первый useEffect
      // Редирект произойдёт через второй useEffect
    } catch (err) {
      console.error("Ошибка подтверждения маппинга:", err);
    }
  }

  function handleReset() {
    dispatch(clearAnalyseState());
    navigate("/new-analyse");
  }

  return (
    <div className="container-page-content">
      <div className="container-header">
        <div className="step-container">
          <div className="step">1</div>
          <h2>Загрузка файла</h2>
        </div>
        <div className="step-container main">
          <div className="step">2</div>
          <h2>Обработка файла</h2>
        </div>
        <div className="step-container">
          <div className="step">3</div>
          <h2>Результат</h2>
        </div>
      </div>

      <div className="container-upload-process">
        <div className="container-header-process">
          <h3>Сопоставление заголовков</h3>
          <p className="description">
            Мы обнаружили, что загруженный файл содержит несколько столбцов.
            Укажите, какие из них соответствуют полям анализа.
          </p>

          {headers && (
            <form className="container-mapping-columns">
              {(
                Object.entries(requiredFields) as [RequiredFieldKey, string][]
              ).map(([key, label]) => (
                <label key={key}>
                  {label}
                  <select
                    value={mapping[key]}
                    onChange={(e) =>
                      setMapping({ ...mapping, [key]: e.target.value })
                    }
                  >
                    <option value="">выбрать</option>
                    {headers.map((h) => (
                      <option key={h} value={h}>
                        {h}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </form>
          )}
        </div>

        {headers && preview && (
          <div className="preview-section">
            <h3>Превью данных</h3>
            <div className="preview-table-wrapper">
              <table className="preview-table">
                <thead>
                  <tr>
                    {headers.map((h) => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, i) => (
                    <tr key={i}>
                      {headers.map((h) => (
                        <td key={h}>{row[h] ?? "—"}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {error && (
          <div className="error-message">
            <p>❌ Ошибка: {error}</p>
          </div>
        )}

        <div className="container-btns">
          <button onClick={handleReset} className="secondary">
            Отменить
          </button>
          <button
            disabled={!mapping.text || loading}
            onClick={handleNext} // 👈 ИСПРАВЛЕНО: теперь вызывается handleNext
            className="primary"
          >
            {loading ? "Обработка..." : "Далее"}
          </button>
        </div>
      </div>
    </div>
  );
}