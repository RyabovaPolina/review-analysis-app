import { Link, useNavigate } from "react-router-dom";
import { loginUser } from "../api/auth";
import { useState } from "react";
import { requestPasswordReset } from "../api/auth";

export default function EmailPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    email: "",
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
      await requestPasswordReset(form.email);
      setSuccess(true);
      navigate('/reset-password?token=');
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
            Почта
            <input
              type="text"
              id="email"
              name="email"
              value={form.email}
              onChange={handleChange}
            />
          </label>
          <button type="submit">Отправить ссылку</button>
        </form>
      </div>
    </>
  );
}
