import { useState } from "react";
import { registerUser } from "../api/auth";
import { Link } from "react-router-dom";

export default function RegisterPage() {
  const [form, setForm] = useState({
    name: "",
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
      await registerUser(form);
      setSuccess(true);
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className="container-page-auth">
      <img src="auth.png" alt="auth" />
      <form className="auth-form" onSubmit={handleSubmit}>
        <label>
          Имя
          <input
            type="text"
            name="name"
            value={form.name}
            onChange={handleChange}
          />
        </label>

        <label>
          Логин (email)
          <input
            type="text"
            name="email"
            value={form.email}
            onChange={handleChange}
          />
        </label>

        <label>
          Пароль
          <input
            type="password"
            name="password"
            value={form.password}
            onChange={handleChange}
          />
        </label>

        {error && <p className="error">{error}</p>}
        {success && <p className="success">Регистрация успешна 🎉</p>}

        <button type="submit">Зарегистрироваться</button>

        <div className="container-btns-link-reg">
          <span>Уже есть аккаунт?</span>
          <Link to="/auth">Войти</Link>
        </div>
      </form>
    </div>
  );
}
