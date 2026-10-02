import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "FinTwin-X · Financial Twin Operating System",
  description:
    "A probabilistic financial intelligence workspace with forecasting, explainable risk, Monte Carlo scenarios and a grounded AI copilot.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="app-main">{children}</main>
        <footer className="site-footer">
          <div>
            <span>FinTwin-X research environment · Synthetic data only</span>
            <span>Model estimates are probabilistic, not financial advice.</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
