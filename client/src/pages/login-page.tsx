import { Link, useNavigate } from "react-router-dom";
import { loginUser } from "../api/auth";
import { useState } from "react";

export default function LoginPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    email: "",
    password: "",
  });

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

    try {
      const res = await loginUser(form);
      setSuccess(true);
      localStorage.setItem("token", res.token);
      navigate("/");
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <>
      <div className="container-page-auth">
        <img src="auth.png" alt="auth" />
        <form className="auth-form" onSubmit={handleSubmit}>
          <label htmlFor="email">
            Логин
            <input
              type="text"
              id="email"
              name="email"
              value={form.email}
              onChange={handleChange}
            />
          </label>
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
          <button type="submit">Войти</button>
          <div className="container-btns-link">
            <Link to="/email-page">Забыли пароль</Link>
            <Link to="/register">Зарегистрироваться</Link>
          </div>
        </form>
      </div>
    </>
  );
}
