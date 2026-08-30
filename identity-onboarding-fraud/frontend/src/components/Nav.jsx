import { NavLink } from "react-router-dom";

const LINKS = [
  { section: "Lab", items: [{ to: "/", label: "Overview", icon: "◆" }] },
  {
    section: "Red Team",
    items: [
      { to: "/red-team", label: "Attack Generator", icon: "⚔" },
      { to: "/attack-lab", label: "Attack Lab", icon: "🗂" },
    ],
  },
  {
    section: "Blue Team",
    items: [
      { to: "/live-kyc", label: "Live KYC Form", icon: "📷" },
      { to: "/simulator", label: "Onboarding Simulator (Auto)", icon: "🧾" },
      { to: "/blue-team", label: "Detection Results", icon: "🛡" },
      { to: "/graph", label: "Identity Graph", icon: "🕸" },
    ],
  },
  { section: "Feedback Loop", items: [{ to: "/closed-loop", label: "Closed Loop", icon: "♻" }] },
];

export default function Nav() {
  return (
    <nav className="sidebar">
      <div className="brand">
        IDENTITY FRAUD
        <br />
        DEFENSE LAB
        <small>Mastercard Innovation Challenge 2026</small>
      </div>
      {LINKS.map((group) => (
        <div key={group.section}>
          <div className="nav-section">{group.section}</div>
          {group.items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
            >
              <span>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
}
