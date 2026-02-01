import { useEffect, useState } from "react";
import { useAppDispatch, useAppSelector } from "../../hooks";
import { setFileAnalyse } from "../../store/slices/analyse-slice";
import "./style.css";
import type { RootState } from "../../store";
import Popup, { type RequiredFieldKey } from "../../components/popup/popup";



export default function NewAnalysePage() {
  const [file, setFile] = useState<File | null>(null);
  const dispatch = useAppDispatch();
  const [showPopup, setShowPopup] = useState(false);

  function handleSubmit() {
    if (!file) return;
    dispatch(setFileAnalyse(file));
  }

  const { headers, preview, loading } = useAppSelector(
    (state: RootState) => state.analyse
  );

  useEffect(() => {
    if (headers && preview) setShowPopup(true);
  }, [headers, preview]);

  function handleNext(data:Record<RequiredFieldKey, string>){
    console.log('next')
  }

  return (
    <>
      <div className="container-page-content">
        <h1>Новый анализ</h1>
        <div className="container-new-analyse">
          <div className="container-new-analyse-img">
            <div className="new-analyse-text-container">
              <span className="new-analyse-text">
                Загрузите <span className="new-analyse-text bold">отзывы</span>{" "}
                клиентов и получите автоматический{" "}
                <span className="new-analyse-text bold">
                  анализ тональности
                </span>
              </span>
            </div>
          </div>
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
            <button onClick={handleSubmit} disabled={!file}>
              Загрузить csv
            </button>
          </div>
        </div>
        {headers && preview && showPopup && <Popup headers={headers} preview={preview} onClose={()=>setShowPopup(false)} onNext={handleNext}/>}
      </div>
    </>
  );
}
