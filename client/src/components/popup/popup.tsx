import { useState } from "react";
import "./style.css";

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
// "text" | "rating"

export default function Popup({
  headers,
  preview,
  onClose,
  onNext,
}: PopupData) {
  const [mapping, setMapping] = useState<Record<RequiredFieldKey, string>>({
    text: "",
    rating: "",
  });

  return (
    <div className="popup-overlay">
      <div className="container-popup">
        <div className="container-popup-header">
          <h2>Сопоставление заголовков</h2>
          <button onClick={onClose}>
            ✕
          </button>
        </div>
        <p className="popup-description">
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
        <button
          disabled={!mapping.text}
          onClick={() => onNext(mapping)}
        >
          Далее
        </button>
      </div>
    </div>
  );
}