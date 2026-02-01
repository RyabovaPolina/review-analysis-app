import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

// ===============================
// ASYNC ACTIONS (thunks)
// ===============================

export const fetchCurrentUser = createAsyncThunk(
  "user/fetchCurrentUser",
  async (_, thunkAPI) => {
    try {
      const res = await fetch("http://localhost:5000/api/user/me", {
        credentials: "include",
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
      });

      if (!res.ok) throw new Error("Не удалось получить пользователя");

      return await res.json();
    } catch (e: any) {
      return thunkAPI.rejectWithValue(e.message);
    }
  }
);

// ===============================
// INITIAL STATE
// ===============================

const initialState = {
  user: null as null | {
    id: string;
    name: string;
    email: string;
    role: string;
  },
  loading: false,
  error: null as string | null,
};

// ===============================
// SLICE
// ===============================

const userSlice = createSlice({
  name: "user",
  initialState,

  reducers: {
    clearSelectedUser(state) {
      state.user = null;
    },

    // Установка информации о текущем пользователе
    setCurrentUser(state, action) {
      state.user = action.payload;
    },
  },

  extraReducers: (builder) => {
    builder
      // -------- Получение всех пользователей --------
      .addCase(fetchCurrentUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchCurrentUser.fulfilled, (state, action) => {
        state.user = action.payload;
        state.loading = false;
      })
      .addCase(fetchCurrentUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  },
});

export const { clearSelectedUser, setCurrentUser } = userSlice.actions;

export default userSlice.reducer;
