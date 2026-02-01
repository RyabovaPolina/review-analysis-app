import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

interface AnalyseUploadResponse {
  message: string;
  headers: string[];
  preview: Record<string, string>[];
}

interface AnalyseState {
  file: File | null;
  loading: boolean;
  error: string | null;
  headers: string[] | null;
  preview: Record<string, string>[] | null;
}

const initialState: AnalyseState = {
  file: null,
  loading: false,
  error: null,
  headers: null,
  preview: null,
};

// ===============================
// ASYNC ACTION
// ===============================

export const setFileAnalyse = createAsyncThunk<
  AnalyseUploadResponse,
  File,
  { rejectValue: string }
>("analyse/setFileAnalyse", async (file, thunkAPI) => {
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

// ===============================
// SLICE
// ===============================

const analyseSlice = createSlice({
  name: "analyse",
  initialState,
  reducers: {
    clearSelectedFile(state) {
      state.file = null;
      state.headers = null;
    },
    setCurrentFile(state, action) {
      state.file = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(setFileAnalyse.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(setFileAnalyse.fulfilled, (state, action) => {
        state.loading = false;
        state.headers = action.payload.headers;
        state.preview = action.payload.preview;
      })
      .addCase(setFileAnalyse.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || "Ошибка";
      });
  },
});

export const { clearSelectedFile, setCurrentFile } = analyseSlice.actions;
export default analyseSlice.reducer;
