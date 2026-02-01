import { Routes, Route } from "react-router-dom";
import AppLayout from "./layout/app-layout";
import LoginPage from "./pages/login-page";
import RegisterPage from "./pages/register-page";
import PersonalCabinet from "./pages/personal-cabinet";
import EmailPage from "./pages/email-page";
import ResetPasswordPage from "./pages/reset-password-page";
import NewAnalysePage from "./pages/new-analyse-page/new-analyse-page";

function App() {
  return (
    <Routes>
      {/* Публичные страницы */}
      <Route path="/auth" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/email-page" element={<EmailPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />

      {/* Приватная часть с навбаром */}
      <Route element={<AppLayout />}>
        <Route path="/" element={<PersonalCabinet />} />
        <Route path="/personal" element={<PersonalCabinet />} />
        <Route path="/new-analyse" element={<NewAnalysePage />} />
        {/* сюда же потом Dashboard, History, NewAnalysis */}
      </Route>
    </Routes>
  );
}

export default App;
