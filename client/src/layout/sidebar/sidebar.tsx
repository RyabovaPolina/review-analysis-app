import {
  X,
  History,
  HomeIcon,
  ChartPie,
  ArrowRight,
} from "lucide-react";
import NavMenuItem from "../../components/nav-menu-item";
import { Link } from "react-router-dom";
import './style.css'

type Props = {
  collapsed: boolean;
  onToggle: (v: boolean) => void;
};

export default function Sidebar({ collapsed, onToggle }: Props) {
  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
      {/* Верх */}
      <div className="sidebar-header">
        <div className="logo">
          {!collapsed && <Link to={"/"}>ReviewAI</Link>}
        </div>

        <button onClick={() => onToggle(!collapsed)}>
          {collapsed ? <ArrowRight size={18} /> : <X size={18} />}
        </button>
      </div>
      <div className="horizontal-line-container">
        <hr />
      </div>

      {/* Навигация */}
      <nav className="sidebar-nav">
        <NavMenuItem
          icon={<HomeIcon size={35} />}
          label="Личный кабинет"
          collapsed={collapsed}
          link={"/personal"}
        />
        <NavMenuItem
          icon={<ChartPie size={35} />}
          label="Новый анализ"
          collapsed={collapsed}
          link={"/new-analyse"}
        />
        <NavMenuItem
          icon={<History size={37} />}
          label="История"
          collapsed={collapsed}
          link={"/personal"}
        />
      </nav>
    </aside>
  );
}
