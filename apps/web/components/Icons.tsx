import type { ReactElement, SVGProps } from "react";

export type IconName =
  | "grid"
  | "pulse"
  | "shield"
  | "nodes"
  | "spark"
  | "server"
  | "wallet"
  | "arrow-up"
  | "arrow-down"
  | "target"
  | "alert"
  | "check"
  | "chevron"
  | "send"
  | "command"
  | "refresh"
  | "menu"
  | "close"
  | "database"
  | "activity"
  | "lock"
  | "clock";

const paths: Record<IconName, ReactElement> = {
  grid: <><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/></>,
  pulse: <path d="M3 12h4l2.2-6 4.2 12 2.1-6H21" />,
  shield: <path d="M12 3 20 6v5c0 5-3.4 8.7-8 10-4.6-1.3-8-5-8-10V6l8-3Zm-3.5 9 2.2 2.2 4.8-5" />,
  nodes: <><circle cx="5" cy="12" r="2.5"/><circle cx="18.5" cy="5" r="2.5"/><circle cx="18.5" cy="19" r="2.5"/><path d="m7.3 10.8 8.8-4.6M7.3 13.2l8.8 4.6"/></>,
  spark: <path d="m12 2 1.6 5.1L19 9l-5.4 1.9L12 16l-1.6-5.1L5 9l5.4-1.9L12 2Zm7 13 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" />,
  server: <><rect x="3" y="4" width="18" height="6" rx="2"/><rect x="3" y="14" width="18" height="6" rx="2"/><path d="M7 7h.01M7 17h.01M11 7h7M11 17h7"/></>,
  wallet: <><path d="M4 6.5A2.5 2.5 0 0 1 6.5 4H18a2 2 0 0 1 2 2v2H6a2 2 0 0 0 0 4h14v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6.5Z"/><path d="M20 8v4h-3a2 2 0 0 1 0-4h3Z"/></>,
  "arrow-up": <path d="m6 15 6-6 6 6M12 9v11" />,
  "arrow-down": <path d="m6 9 6 6 6-6M12 4v11" />,
  target: <><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M15 9 21 3M17 3h4v4"/></>,
  alert: <><path d="M10.3 4.1 2.5 18a2 2 0 0 0 1.8 3h15.4a2 2 0 0 0 1.8-3L13.7 4.1a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/></>,
  check: <path d="m5 12 4 4L19 6" />,
  chevron: <path d="m9 18 6-6-6-6" />,
  send: <><path d="m22 2-7 20-4-9-9-4 20-7Z"/><path d="M22 2 11 13"/></>,
  command: <><rect x="4" y="4" width="16" height="16" rx="4"/><path d="m8 9 3 3-3 3M13 15h3"/></>,
  refresh: <><path d="M20 11a8 8 0 0 0-14.9-4M4 4v5h5"/><path d="M4 13a8 8 0 0 0 14.9 4M20 20v-5h-5"/></>,
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/></>,
  activity: <><path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/></>,
  lock: <><rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/></>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
};

export default function Icon({ name, ...props }: { name: IconName } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name]}
    </svg>
  );
}

