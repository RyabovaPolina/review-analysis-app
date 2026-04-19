import express from "express";
import multer from "multer";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import path from "path";
import fs from "fs";
import csvParser from "csv-parser";
import { pool } from "../db";
import dotenv from "dotenv";
import { spawn } from "child_process";
import { fileURLToPath } from "url";

// Получаем __dirname для ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, "../../.env") });

const router = express.Router();
const upload = multer({ dest: "uploads/" });

const s3Client = new S3Client({
  endpoint: process.env.S3_ENDPOINT,
  region: "ru-central1",
  credentials: {
    accessKeyId: process.env.S3_ACCESS_KEY!,
    secretAccessKey: process.env.S3_SECRET_KEY!,
  },
});

const bucketName = process.env.S3_BUCKET_NAME!;

function detectSeparator(line: string): string {
  const separators = [",", ";", "\t", "|"];
  let maxCount = 0;
  let detected = ",";

  for (const sep of separators) {
    const count = line.split(sep).length;
    if (count > maxCount) {
      maxCount = count;
      detected = sep;
    }
  }
  return detected;
}

// 1️⃣ Загрузка и предпросмотр файла
router.post("/upload", upload.single("csv"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "Файл не загружен" });
  }

  const filePath = req.file.path;
  const userId = req.body.userId || 1;

  try {
    const firstLine = fs.readFileSync(filePath, "utf8").split("\n")[0];
    const separator = detectSeparator(firstLine);

    const rows: any[] = [];
    await new Promise<void>((resolve, reject) => {
      fs.createReadStream(filePath)
        .pipe(csvParser({ separator }))
        .on("data", (row) => rows.push(row))
        .on("end", resolve)
        .on("error", reject);
    });

    const headers = rows.length ? Object.keys(rows[0]) : [];

    const result = await pool.query(
      `INSERT INTO uploaded_files 
       (user_id, original_filename, file_size, separator, headers, status)
       VALUES ($1, $2, $3, $4, $5, $6)
       RETURNING id`,
      [
        userId,
        req.file.originalname,
        req.file.size,
        separator,
        JSON.stringify(headers),
        "pending",
      ]
    );

    const tempFileId = result.rows[0].id;

    // ИСПРАВЛЕНО: убрали дублирование "server"
    const tempFilePath = path.join("uploads", `${tempFileId}.csv`);

    // Создаём папку uploads если её нет
    if (!fs.existsSync("uploads")) {
      fs.mkdirSync("uploads", { recursive: true });
    }

    fs.renameSync(filePath, tempFilePath);

    res.json({
      message: "Файл загружен для предпросмотра",
      fileId: tempFileId,
      separator,
      headers,
      preview: rows.slice(0, 5),
    });
  } catch (error) {
    console.error(error);
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
    res.status(500).json({ message: "Ошибка загрузки файла" });
  }
});

// 2️⃣ Подтверждение маппинга и загрузка в S3
router.post("/confirm-mapping/:fileId", async (req, res) => {
  const { fileId } = req.params;
  const { mapping } = req.body;

  try {
    const fileResult = await pool.query(
      "SELECT * FROM uploaded_files WHERE id = $1 AND status = $2",
      [fileId, "pending"]
    );

    if (fileResult.rows.length === 0) {
      return res
        .status(404)
        .json({ message: "Файл не найден или уже обработан" });
    }

    const fileInfo = fileResult.rows[0];

    // ИСПРАВЛЕНО: убрали дублирование "server"
    const tempFilePath = path.join("uploads", `${fileId}.csv`);

    console.log("📂 Ищем файл:", tempFilePath);

    if (!fs.existsSync(tempFilePath)) {
      return res.status(404).json({ message: "Временный файл не найден" });
    }

    const separator = fileInfo.separator;
    const rows: any[] = [];

    await new Promise<void>((resolve, reject) => {
      fs.createReadStream(tempFilePath)
        .pipe(csvParser({ separator }))
        .on("data", (row) => rows.push(row))
        .on("end", resolve)
        .on("error", reject);
    });

    const renamedRows = rows.map((row) => {
      const newRow: any = {};

      for (const [standardName, originalName] of Object.entries(mapping)) {
        if (originalName && row[originalName as string] !== undefined) {
          newRow[standardName] = row[originalName as string];
        }
      }

      for (const [key, value] of Object.entries(row)) {
        if (!Object.values(mapping).includes(key)) {
          newRow[key] = value;
        }
      }

      return newRow;
    });

    // В функции confirm-mapping, где конвертируем в CSV:
    const headers = Object.keys(renamedRows[0]);

    // Правильное экранирование CSV
    const csvContent = [
      headers.map((h) => `"${h}"`).join(","), // заголовки всегда в кавычках
      ...renamedRows.map((row) =>
        headers
          .map((h) => {
            const value = row[h] || "";
            // Экранируем кавычки внутри значений
            const escaped = String(value).replace(/"/g, '""');
            return `"${escaped}"`;
          })
          .join(",")
      ),
    ].join("\n");

    const timestamp = Date.now();
    const userId = fileInfo.user_id;
    const s3Key = `uploads/${userId}/${timestamp}_${fileInfo.original_filename}`;

    await s3Client.send(
      new PutObjectCommand({
        Bucket: bucketName,
        Key: s3Key,
        Body: Buffer.from(csvContent, "utf-8"),
        ContentType: "text/csv",
      })
    );

    await pool.query(
      `UPDATE uploaded_files 
       SET s3_key = $1, 
           column_mapping = $2, 
           status = $3,
           headers = $4
       WHERE id = $5`,
      [
        s3Key,
        JSON.stringify(mapping),
        "uploaded",
        JSON.stringify(headers),
        fileId,
      ]
    );

    fs.unlinkSync(tempFilePath);

    res.json({
      message: "Файл успешно обработан и загружен",
      fileId,
      s3Key,
      mapping,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Ошибка обработки маппинга" });
  }
});

// 3️⃣ Запуск анализа
router.post("/analyze/:fileId", async (req, res) => {
  const { fileId } = req.params;

  try {
    const fileResult = await pool.query(
      "SELECT s3_key, status, column_mapping FROM uploaded_files WHERE id = $1",
      [fileId]
    );

    if (fileResult.rows.length === 0) {
      return res.status(404).json({ message: "Файл не найден" });
    }

    const { s3_key, status } = fileResult.rows[0];

    if (status !== "uploaded") {
      return res.status(400).json({ message: "Файл не готов к анализу" });
    }

    await pool.query("UPDATE uploaded_files SET status = $1 WHERE id = $2", [
      "processing",
      fileId,
    ]);

    // ИСПРАВЛЕНО: используем __dirname для правильного пути
    // Полный путь к Python из venv
    const pythonPath = path.join(
      __dirname,
      "../../python/venv/Scripts/python.exe"
    );

    const pythonScriptPath = path.join(__dirname, "../../python/analysis.py");

    console.log("🐍 Python script path:", pythonScriptPath);
    console.log("📂 File exists:", fs.existsSync(pythonScriptPath));
    console.log("📂 Current working dir:", process.cwd());

    if (!fs.existsSync(pythonScriptPath)) {
      await pool.query("UPDATE uploaded_files SET status = $1 WHERE id = $2", [
        "failed",
        fileId,
      ]);
      return res.status(500).json({
        message: "Python скрипт не найден",
        path: pythonScriptPath,
      });
    }
    const pythonProcess = spawn(
      pythonPath,
      [pythonScriptPath, s3_key, "text"],
      {
        shell: false,
        cwd: path.join(__dirname, "../.."), // корень проекта
        env: { ...process.env },
      }
    );

    let pythonOutput = "";
    let pythonError = "";

    pythonProcess.stdout.on("data", (data: Buffer) => {
      const output = data.toString();
      console.log("Python stdout:", output);
      pythonOutput += output;
    });

    pythonProcess.stderr.on("data", (data: Buffer) => {
      const error = data.toString();
      console.error("Python stderr:", error);
      pythonError += error;
    });

    pythonProcess.on("error", (error) => {
      console.error("Ошибка запуска Python:", error);
    });

    pythonProcess.on("close", async (code: number) => {
      console.log(`Python процесс завершился с кодом ${code}`);

      if (code !== 0) {
        console.error("Python error:", pythonError);
        await pool.query(
          "UPDATE uploaded_files SET status = $1 WHERE id = $2",
          ["failed", fileId]
        );
        return res.status(500).json({
          message: "Ошибка анализа",
          error: pythonError || "Python завершился с ошибкой",
        });
      }

      try {
        const result = JSON.parse(pythonOutput);

        await pool.query(
          `INSERT INTO analysis_results 
           (file_id, s3_result_key, positive_count, negative_count, neutral_count, total_reviews, avg_sentiment_score)
           VALUES ($1, $2, $3, $4, $5, $6, $7)`,
          [
            fileId,
            result.result_key,
            result.positive_count,
            result.negative_count,
            result.neutral_count,
            result.total_reviews,
            result.avg_sentiment_score,
          ]
        );

        await pool.query(
          "UPDATE uploaded_files SET status = $1 WHERE id = $2",
          ["completed", fileId]
        );

        res.json({
          message: "Анализ завершен",
          results: result,
        });
      } catch (error) {
        console.error("Ошибка парсинга результатов:", error);
        await pool.query(
          "UPDATE uploaded_files SET status = $1 WHERE id = $2",
          ["failed", fileId]
        );
        res.status(500).json({
          message: "Ошибка обработки результатов",
          output: pythonOutput,
        });
      }
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Ошибка запуска анализа" });
  }
});

// 4️⃣ Получение результатов
router.get("/results/:fileId", async (req, res) => {
  const { fileId } = req.params;

  try {
    const result = await pool.query(
      `SELECT ar.*, uf.original_filename, uf.upload_date, uf.column_mapping
       FROM analysis_results ar
       JOIN uploaded_files uf ON ar.file_id = uf.id
       WHERE ar.file_id = $1`,
      [fileId]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ message: "Результаты не найдены" });
    }

    res.json(result.rows[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Ошибка получения результатов" });
  }
});

export default router;
