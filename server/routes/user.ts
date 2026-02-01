import { Router } from 'express';
import { authMiddleware } from '../middleware/auth';
import { pool } from '../db';

const router = Router();

router.get("/me", authMiddleware, async (req, res) => {
  try {
    const userId = req.user!.userId;

    const result = await pool.query(
      "SELECT id, name, email, role FROM users WHERE id = $1",
      [userId]
    );

    if (!result.rows.length) {
      return res.status(404).json({ message: "Пользователь не найден" });
    }

    res.json(result.rows[0]);
  } catch (e) {
    res.status(500).json({ message: "Ошибка сервера" });
  }
});


export default router;
