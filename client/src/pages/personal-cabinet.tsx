import type { RootState } from "../store";
import { useEffect } from "react";
import { fetchCurrentUser } from "../store/slices/user-slice";
import { useAppDispatch, useAppSelector } from "../hooks";

export default function PersonalCabinet() {
  const dispatch = useAppDispatch();
  const { user, loading, error } = useAppSelector(
    (state: RootState) => state.user
  );

  useEffect(() => {
    dispatch(fetchCurrentUser());
  }, [dispatch]);

  if (!user) return <div>Загрузка...</div>;

  if (error) return <div>{error}</div>;
  if (!user) return null;

  const { name, email, role } = user;

  return (
    <div className="container-page-content">
      <h1>Личный кабинет</h1>
      <div className="profile-grid">
        <section className="card">
          <h3>Пользователь</h3>
          <div className="info-card">
            <div className="image-card">
              <img src="auth.png" alt="" />
            </div>
            <div className="info-container">
              <p>
                <b>Имя:</b> {name}
              </p>
              <p>
                <b>Email:</b> {email}
              </p>
              <p>
                <b>Роль:</b> {role}
              </p>
            </div>
          </div>
        </section>

        <section className="card">
          <h3>Статистика</h3>
          <div className="info-card">
            <div className="stat-info">
              <p>Анализов: 12</p>
            </div>
            <div className="stat-info">
              <p>Отзывов: 840</p>
            </div>
            <div className="stat-info">
              <p>Позитив: 56%</p>
            </div>
          </div>
        </section>

        <section className="card">
          <h3>Последние анализы</h3>
          <div className="analyse-list-container">
            <div className="analyse">
              <p>12.01.2026 — CSV — 120 отзывов</p>
              <button>Посмотреть</button>
            </div>
            <div className="analyse">
              <p>09.01.2026 — Текст — 45 отзывов</p>
              <button>Посмотреть</button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
