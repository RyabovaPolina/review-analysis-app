import express from "express";
import fs from "fs";
import csvParser from "csv-parser";
import multer from "multer";

const router = express.Router();
const upload = multer({ dest: "uploads/" });

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


router.post("/upload", upload.single("csv"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ message: "Файл не загружен" });
  }

  const filePath = req.file.path;

  // 1️⃣ читаем первую строку
  const firstLine = fs
    .readFileSync(filePath, "utf8")
    .split("\n")[0];

  // 2️⃣ определяем разделитель
  const separator = detectSeparator(firstLine);

  const rows: any[] = [];

  fs.createReadStream(filePath)
    .pipe(csvParser({ separator }))
    .on("data", (row) => rows.push(row))
    .on("end", () => {
      const headers = rows.length ? Object.keys(rows[0]) : [];

      fs.unlinkSync(filePath);

      res.json({
        message: "Файл загружен",
        separator,
        headers,
        preview: rows.slice(0, 5),
      });
    });
});

export default router;
