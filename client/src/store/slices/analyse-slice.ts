import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

interface AnalyseUploadResponse {
  message: string;
  fileId: number;
  headers: string[];
  preview: Record<string, string>[];
}

interface ConfirmMappingResponse {
  message: string;
  fileId: number;
  s3Key: string;
  mapping: Record<string, string>;
}

interface AnalyzeResponse {
  message: string;
  results: {
    positive_count: number;
    negative_count: number;
    neutral_count: number;
    total_reviews: number;
    avg_sentiment_score: number;
    result_key: string;
  };
}

interface AnalyseState {
  file: File | null;
  fileId: number | null;
  loading: boolean;
  error: string | null;
  headers: string[] | null;
  preview: Record<string, string>[] | null;
  mappingConfirmed: boolean;
  analyzing: boolean; // 👈 новое
  analysisResults: AnalyzeResponse['results'] | null; // 👈 новое
}

const initialState: AnalyseState = {
  file: null,
  fileId: null,
  loading: false,
  error: null,
  headers: null,
  preview: null,
  mappingConfirmed: false,
  analyzing: false,
  analysisResults: null,
};

// Загрузка файла для предпросмотра
export const uploadFilePreview = createAsyncThunk<
  AnalyseUploadResponse,
  File,
  { rejectValue: string }
>("analyse/uploadFilePreview", async (file, thunkAPI) => {
  try {
    const formData = new FormData();
    formData.append("csv", file);

    const res = await fetch("http://localhost:5000/api/analyse/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || "Ошибка при загрузке файла");
    }

    return await res.json();
  } catch (e: any) {
    return thunkAPI.rejectWithValue(e.message);
  }
});

// Подтверждение маппинга колонок
export const confirmColumnMapping = createAsyncThunk<
  ConfirmMappingResponse,
  { fileId: number; mapping: Record<string, string> },
  { rejectValue: string }
>("analyse/confirmMapping", async ({ fileId, mapping }, thunkAPI) => {
  try {
    const res = await fetch(
      `http://localhost:5000/api/analyse/confirm-mapping/${fileId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mapping }),
      }
    );

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || "Ошибка подтверждения маппинга");
    }

    return await res.json();
  } catch (e: any) {
    return thunkAPI.rejectWithValue(e.message);
  }
});

// 👇 НОВЫЙ: Запуск анализа
export const startAnalysis = createAsyncThunk<
  AnalyzeResponse,
  number, // fileId
  { rejectValue: string }
>("analyse/startAnalysis", async (fileId, thunkAPI) => {
  try {
    const res = await fetch(
      `http://localhost:5000/api/analyse/analyze/${fileId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      }
    );

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || "Ошибка запуска анализа");
    }

    return await res.json();
  } catch (e: any) {
    return thunkAPI.rejectWithValue(e.message);
  }
});

const analyseSlice = createSlice({
  name: "analyse",
  initialState,
  reducers: {
    clearAnalyseState(state) {
      state.file = null;
      state.fileId = null;
      state.headers = null;
      state.preview = null;
      state.mappingConfirmed = false;
      state.error = null;
      state.analyzing = false;
      state.analysisResults = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Загрузка для предпросмотра
      .addCase(uploadFilePreview.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(uploadFilePreview.fulfilled, (state, action) => {
        state.loading = false;
        state.fileId = action.payload.fileId;
        state.headers = action.payload.headers;
        state.preview = action.payload.preview;
      })
      .addCase(uploadFilePreview.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || "Ошибка";
      })
      // Подтверждение маппинга
      .addCase(confirmColumnMapping.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(confirmColumnMapping.fulfilled, (state) => {
        state.loading = false;
        state.mappingConfirmed = true;
      })
      .addCase(confirmColumnMapping.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || "Ошибка";
      })
      // 👇 НОВЫЙ: Анализ
      .addCase(startAnalysis.pending, (state) => {
        state.analyzing = true;
        state.error = null;
      })
      .addCase(startAnalysis.fulfilled, (state, action) => {
        state.analyzing = false;
        state.analysisResults = action.payload.results;
      })
      .addCase(startAnalysis.rejected, (state, action) => {
        state.analyzing = false;
        state.error = action.payload || "Ошибка анализа";
      });
  },
});

export const { clearAnalyseState } = analyseSlice.actions;
export default analyseSlice.reducer;