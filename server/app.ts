import express from 'express';
import cors from 'cors';
import authRoutes from './routes/auth';
import userRoutes from './routes/user';
import analyseRoutes from './routes/analyse';

const app = express();

app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true,
}));

app.use(express.json());
app.use('/api/auth', authRoutes);
app.use('/api/user', userRoutes);
app.use('/api/analyse', analyseRoutes);

app.listen(5000, () => {
  console.log('Server started on http://localhost:5000');
});
