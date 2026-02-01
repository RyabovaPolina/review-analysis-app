import { Router } from 'express';
import type { Request, Response } from 'express';
import bcrypt from 'bcrypt';
import { pool } from '../db';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';

const router = Router();
const SALT_ROUNDS = 10;

router.post('/register', async (req: Request, res: Response) => {
  try {
    const { name, email, password } = req.body;

    // 1. Валидация входных данных
    if (!email || !password) {
      return res.status(400).json({
        message: 'Email и пароль обязательны',
      });
    }

    // 2. Проверка существования пользователя
    const existingUser = await pool.query(
      'SELECT id FROM public.users WHERE email = $1',
      [email]
    );

    if (existingUser.rowCount && existingUser.rowCount > 0) {
      return res.status(409).json({
        message: 'Пользователь с таким email уже существует',
      });
    }

    // 3. Хэширование пароля
    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);

    // 4. Создание пользователя
    const result = await pool.query(
      `
      INSERT INTO public.users (
        email,
        password_hash,
        name,
        role,
        is_active
      )
      VALUES ($1, $2, $3, 'user', TRUE)
      RETURNING
        id,
        email,
        name,
        role,
        is_active,
        created_at
      `,
      [email, passwordHash, name ?? null]
    );

    // 5. Ответ клиенту
    return res.status(201).json({
      message: 'Пользователь успешно зарегистрирован',
      user: result.rows[0],
    });

  } catch (error: any) {
    console.error('REGISTER ERROR:', error);

    return res.status(500).json({
      message: 'Внутренняя ошибка сервера',
    });
  }
});


const JWT_SECRET = process.env.JWT_SECRET || 'dev_secret';
const JWT_EXPIRES_IN = '7d';

router.post('/login', async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;

    // 1. Проверка входных данных
    if (!email || !password) {
      return res.status(400).json({
        message: 'Email и пароль обязательны',
      });
    }

    // 2. Поиск пользователя
    const result = await pool.query(
      `
      SELECT id, email, password_hash, name, role, is_active
      FROM public.users
      WHERE email = $1
      `,
      [email]
    );

    if (result.rowCount === 0) {
      return res.status(401).json({
        message: 'Неверный email или пароль',
      });
    }

    const user = result.rows[0];

    // 3. Проверка активности
    if (!user.is_active) {
      return res.status(403).json({
        message: 'Аккаунт заблокирован',
      });
    }

    // 4. Сравнение паролей
    const isPasswordValid = await bcrypt.compare(
      password,
      user.password_hash
    );

    if (!isPasswordValid) {
      return res.status(401).json({
        message: 'Неверный email или пароль',
      });
    }

    // 5. Генерация JWT
    const token = jwt.sign(
      {
        userId: user.id,
        role: user.role,
      },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRES_IN }
    );

    // 6. Ответ клиенту
    return res.json({
      message: 'Успешный вход',
      token,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
      },
    });

  } catch (error) {
    console.error('LOGIN ERROR:', error);
    return res.status(500).json({
      message: 'Внутренняя ошибка сервера',
    });
  }
});

router.post('/forgot-password', async (req, res) => {
  const { email } = req.body;

  if (!email) {
    return res.status(400).json({ message: 'Email обязателен' });
  }

  const userRes = await pool.query(
    'SELECT id FROM public.users WHERE email = $1',
    [email]
  );

  if (userRes.rowCount === 0) {
    return res.json({
      message: 'Если email существует, инструкция будет отправлена',
    });
  }

  const userId = userRes.rows[0].id;

  // 1. Генерация токена
  const token = crypto.randomBytes(32).toString('hex');

  // 2. Хеш токена
  const tokenHash = await bcrypt.hash(token, 10);

  // 3. Срок жизни (например 1 час)
  const expiresAt = new Date(Date.now() + 60 * 60 * 1000);

  await pool.query(
    `
    INSERT INTO public.password_reset_tokens
      (user_id, token_hash, expires_at)
    VALUES ($1, $2, $3)
    `,
    [userId, tokenHash, expiresAt]
  );

  // 4. Ссылка (пока просто лог)
  const resetLink = `http://localhost:5173/reset-password?token=${token}`;

  console.log('RESET LINK:', resetLink);

  return res.json({
    message: 'Если email существует, инструкция будет отправлена',
  });
});

router.post('/reset-password', async (req, res) => {
  const { token, newPassword } = req.body;

  if (!token || !newPassword) {
    return res.status(400).json({ message: 'Некорректные данные' });
  }

  // 1. Получаем все активные токены
  const tokensRes = await pool.query(
    `
    SELECT id, user_id, token_hash, expires_at
    FROM public.password_reset_tokens
    WHERE used = FALSE AND expires_at > now()
    `
  );

  // 2. Ищем подходящий
  let matchedToken = null;

  for (const row of tokensRes.rows) {
    const isValid = await bcrypt.compare(token, row.token_hash);
    if (isValid) {
      matchedToken = row;
      break;
    }
  }

  if (!matchedToken) {
    return res.status(400).json({ message: 'Токен недействителен' });
  }

  // 3. Хеш нового пароля
  const newPasswordHash = await bcrypt.hash(newPassword, 10);

  // 4. Обновляем пароль
  await pool.query(
    `
    UPDATE public.users
    SET password_hash = $1
    WHERE id = $2
    `,
    [newPasswordHash, matchedToken.user_id]
  );

  // 5. Помечаем токен использованным
  await pool.query(
    `
    UPDATE public.password_reset_tokens
    SET used = TRUE
    WHERE id = $1
    `,
    [matchedToken.id]
  );

  return res.json({ message: 'Пароль успешно обновлён' });
});


export default router;
