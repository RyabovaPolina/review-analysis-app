// layout/Header.tsx
import { User } from 'lucide-react';

export default function Header() {
  return (
    <header className="header">
      <div />
      <div className="user-info">
        <User size={25} />
        <h3>Поля Рябова</h3>
      </div>
    </header>
  );
}
