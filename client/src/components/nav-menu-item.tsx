import { Link } from "react-router-dom";

type ItemProps = {
  icon: React.ReactNode;
  label: string;
  collapsed: boolean;
  link: string,
};

export default function NavMenuItem({ icon, label, collapsed, link }: ItemProps) {
  return (
    <div className="sidebar-item">
      {icon}
      {!collapsed && <Link to={link}>{label}</Link>}
    </div>
  );
}
