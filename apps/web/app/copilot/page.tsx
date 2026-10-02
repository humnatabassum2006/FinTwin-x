"use client";

import { useState } from "react";
import CopilotChat from "@/components/CopilotChat";
import { PageHeader, StatusPill } from "@/components/UI";

export default function CopilotPage() {
  const [userId, setUserId] = useState(8);
  return (
    <>
      <PageHeader
        eyebrow="Agentic decision support"
        title="Ask your financial twin."
        description="Get evidence-backed answers from a controlled agent that delegates calculations to analytics, ML and simulation tools."
        userId={userId}
        onUserChange={setUserId}
      >
        <StatusPill tone="good">Tool-grounded</StatusPill>
      </PageHeader>
      <div className="content-stack"><CopilotChat key={userId} userId={userId} /></div>
    </>
  );
}
