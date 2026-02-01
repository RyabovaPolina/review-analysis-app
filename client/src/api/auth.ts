// api/auth.ts
export async function registerUser(data: {
  name: string;
  email: string;
  password: string;
}) {
  const res = await fetch("http://localhost:5000/api/auth/register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || "Ошибка регистрации");
  }

  return res.json();
}

export async function loginUser(data: { email: string; password: string }) {
  const res = await fetch("http://localhost:5000/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message);
  }

  return res.json();
}

// api/auth.ts
export const requestPasswordReset = (email: string) =>
  fetch("http://localhost:5000/api/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
    headers: { "Content-Type": "application/json" },
  });

export const resetPassword = (data: { token: string; newPassword: string }) =>
  fetch("http://localhost:5000/api/auth/reset-password", {
    method: "POST",
    body: JSON.stringify(data),
    headers: { "Content-Type": "application/json" },
  });
