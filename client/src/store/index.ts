import { configureStore } from "@reduxjs/toolkit";
import userReducer from './slices/user-slice'
import analyseReducer from './slices/analyse-slice'

const store = configureStore({
    reducer:{
        user:userReducer,
        analyse: analyseReducer,
    }
})

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store