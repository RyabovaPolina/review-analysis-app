import { Pool } from 'pg';

const config = {
  user: 'postgres',
  password: 'postgres',
  host: 'localhost',
  port: 5432,
  database: 'db',
};

export const pool = new Pool(config);
