import { Router } from 'express';
import { authMiddleware } from '../middleware/auth';

const router = Router();

router.get('/profile', authMiddleware, (req, res) => {
  res.json({
    message: 'Доступ разрешён',
  });
});

export default router;
