import { useNavigate } from "react-router-dom";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { resetPassword } from "../api/auth";

export default function ResetPasswordPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    password: "",
    confirmPassword: "",
  });
  const [params] = useSearchParams();
  const token = params.get("token");

  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (form.password !== form.confirmPassword) {
      setError("Пароли не совпадают");
      return;
    }
    if (!token) {
      setError("Ссылка для восстановления недействительна");
      return;
    }

    try {
      await resetPassword({
        token,
        newPassword: form.password,
      });

      navigate("/auth");
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <>
      <div className="container-page-auth">
        <img src="auth.png" alt="auth" />
        <form className="auth-form" onSubmit={handleSubmit}>
          <label htmlFor="password">
            Пароль
            <input
              type="password"
              id="password"
              name="password"
              value={form.password}
              onChange={handleChange}
            />
          </label>
          <label htmlFor="confirmPassword">
            Повторите пароль
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={form.confirmPassword}
              onChange={handleChange}
            />
          </label>
          <button type="submit">Сохранить изменения</button>
        </form>
      </div>
    </>
  );
}
