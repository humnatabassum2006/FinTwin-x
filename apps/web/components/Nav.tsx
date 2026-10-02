"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import Icon, { type IconName } from "./Icons";
import { api } from "@/lib/api";

const LINKS: { href: string; label: string; icon: IconName; fresh?: boolean }[] = [
  { href: "/", label: "Command center", icon: "grid" },
  { href: "/intelligence", label: "Cash intelligence", icon: "pulse", fresh: true },
  { href: "/risk", label: "Risk engine", icon: "shield", fresh: true },
  { href: "/scenarios", label: "Scenario lab", icon: "nodes" },
  { href: "/copilot", label: "AI copilot", icon: "spark" },
  { href: "/models", label: "Model operations", icon: "server", fresh: true },
];

function Brand() {
  return (
    <Link href="/" className="brand" aria-label="FinTwin-X command center">
      <span className="brand-mark" aria-hidden="true" />
      <span>
        <span className="brand-name">FinTwin<em>·X</em></span>
        <span className="brand-sub">Financial twin OS</span>
      </span>
    </Link>
  );
}

function NavLinks({ mobile = false }: { mobile?: boolean }) {
  const path = usePathname();
  return (
    <nav className={mobile ? "mobile-nav" : "nav-list"} aria-label="Primary navigation">
      {LINKS.map((item) => {
        const active = item.href === "/" ? path === "/" : path.startsWith(item.href);
        return (
          <Link key={item.href} href={item.href} className={`nav-link ${active ? "active" : ""}`}>
            <Icon name={item.icon} />
            <span>{item.label}</span>
            {item.fresh && <span className="nav-new">New</span>}
          </Link>
        );
      })}
    </nav>
  );
}

export default function Nav() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api.health().then((response) => setOnline(response.status === "ok")).catch(() => setOnline(false));
  }, []);

  return (
    <>
      <aside className="sidebar">
        <Brand />
        <div className="sidebar-context">
          <span>Active environment</span>
          <strong>Pakistan · PKR network</strong>
          <small>Synthetic data environment</small>
        </div>
        <div className="nav-label">Workspace</div>
        <NavLinks />
        <div className="sidebar-footer">
          <div className={`system-state ${online === false ? "system-offline" : online == null ? "system-checking" : ""}`}>{online == null ? "Checking engines" : online ? "All engines operational" : "API connection offline"}</div>
          <p>Forecast · Risk · Simulation · RAG</p>
        </div>
      </aside>

      <header className="mobile-header">
        <div className="mobile-bar">
          <Brand />
          <span className={`system-state ${online === false ? "system-offline" : online == null ? "system-checking" : ""}`}>{online ? "Live" : online === false ? "Offline" : "Check"}</span>
        </div>
        <NavLinks mobile />
      </header>
    </>
  );
}
