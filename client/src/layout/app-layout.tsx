// layout/AppLayout.tsx
import { Outlet } from "react-router-dom";
import { useState } from "react";
import Sidebar from "./sidebar/sidebar";
import Header from "./header";
import "../App.css";

export default function AppLayout() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="app-layout">
      <Sidebar collapsed={collapsed} onToggle={setCollapsed} />

      <div className="main-area">
        <Header />
        <main className="content">
          <Outlet>
          </Outlet>
        </main>
      </div>
    </div>
  );
}
