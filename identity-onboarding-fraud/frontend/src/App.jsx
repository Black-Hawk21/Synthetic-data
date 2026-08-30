import { Routes, Route } from "react-router-dom";
import Nav from "./components/Nav.jsx";
import Overview from "./pages/Overview.jsx";
import RedTeam from "./pages/RedTeam.jsx";
import LiveOnboarding from "./pages/LiveOnboarding.jsx";
import OnboardingSimulator from "./pages/OnboardingSimulator.jsx";
import BlueTeam from "./pages/BlueTeam.jsx";
import IdentityGraph from "./pages/IdentityGraph.jsx";
import AttackLab from "./pages/AttackLab.jsx";
import ClosedLoop from "./pages/ClosedLoop.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <Nav />
      <main className="main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/red-team" element={<RedTeam />} />
          <Route path="/live-kyc" element={<LiveOnboarding />} />
          <Route path="/simulator" element={<OnboardingSimulator />} />
          <Route path="/blue-team" element={<BlueTeam />} />
          <Route path="/graph" element={<IdentityGraph />} />
          <Route path="/attack-lab" element={<AttackLab />} />
          <Route path="/closed-loop" element={<ClosedLoop />} />
        </Routes>
      </main>
    </div>
  );
}
