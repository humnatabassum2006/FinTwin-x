"use client";

import { useEffect, useState } from "react";
import ScenarioLab from "@/components/ScenarioLab";
import { api } from "@/lib/api";
import { money, pct } from "@/lib/format";
import { MetricCard, PageHeader } from "@/components/UI";

export default function ScenariosPage() {
  const [userId, setUserId] = useState(8);
  const [goal, setGoal] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    api.goal(userId).then(setGoal).catch(() => setGoal(null));
  }, [userId]);

  const probability = Number(goal?.probability ?? 0);
  const expected = Number(goal?.expected_amount ?? 0);
  const required = Number(goal?.required_monthly ?? 0);
  const shortfall = Number(goal?.shortfall ?? 0);

  return (
    <>
      <PageHeader
        eyebrow="Monte Carlo decision lab"
        title="Test the decision before life does."
        description="Change income, inflation, debt and investing assumptions, then compare thousands of plausible financial futures."
        userId={userId}
        onUserChange={setUserId}
      />
      <div className="content-stack">
        {goal && (
          <div className="grid-4">
            <MetricCard label="Goal success chance" value={pct(probability)} hint="Current trajectory" icon="target" tone={probability >= .7 ? "green" : "amber"} />
            <MetricCard label="Expected goal value" value={money(expected)} hint="Across simulated paths" icon="spark" tone="cyan" />
            <MetricCard label="Required each month" value={money(required)} hint="To stay on target" icon="wallet" tone="violet" />
            <MetricCard label="Median shortfall" value={money(shortfall)} hint="At current contribution" icon="alert" tone={shortfall > 0 ? "rose" : "green"} />
          </div>
        )}
        <ScenarioLab userId={userId} />
      </div>
    </>
  );
}
