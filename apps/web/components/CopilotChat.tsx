"use client";

import { useRef, useState } from "react";
import { api, type AgentResponse } from "@/lib/api";
import Icon from "./Icons";
import { Panel, StatusPill } from "./UI";

const EXAMPLES = [
  { title: "Affordability", text: "Can I afford a Rs 4,000,000 car?", icon: "wallet" as const },
  { title: "Income shock", text: "What happens if my income falls by 20%?", icon: "pulse" as const },
  { title: "Explain risk", text: "Why is my risk score what it is?", icon: "shield" as const },
  { title: "Goal outlook", text: "How likely am I to reach my primary goal?", icon: "target" as const },
  { title: "Optimise", text: "What is the safest way to reach my goal?", icon: "nodes" as const },
  { title: "Anomalies", text: "Which spending activity looks unusual?", icon: "alert" as const },
];

function InlineText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return <>{parts.map((part, index) => part.startsWith("**") && part.endsWith("**") ? <strong key={index}>{part.slice(2, -2)}</strong> : <span key={index}>{part}</span>)}</>;
}

function SafeAnswer({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <div className="space-y-2">
      {lines.filter(Boolean).map((line, index) => {
        const bullet = /^[-*]\s+/.test(line);
        return <p key={index} className={bullet ? "pl-3 before:mr-2 before:text-[#5cf2cc] before:content-['•']" : ""}><InlineText text={line.replace(/^[-*]\s+/, "")} /></p>;
      })}
    </div>
  );
}

type Message = { role: "user" | "assistant"; text: string; meta?: AgentResponse };

export default function CopilotChat({ userId }: { userId: number }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [showTrace, setShowTrace] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  const ask = async (text: string) => {
    const clean = text.trim();
    if (!clean || busy) return;
    setMessages((current) => [...current, { role: "user", text: clean }]);
    setQuestion("");
    setBusy(true);
    try {
      const response = await api.chat(userId, clean);
      setMessages((current) => [...current, { role: "assistant", text: response.answer, meta: response }]);
    } catch (requestError) {
      setMessages((current) => [...current, { role: "assistant", text: `The copilot could not complete this request: ${(requestError as Error).message}` }]);
    } finally {
      setBusy(false);
      window.setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  return (
    <div className="grid-3">
      <Panel className="span-2 !p-0" eyebrow="" title="">
        <div className="flex min-h-[620px] flex-col">
          <div className="flex flex-wrap items-center gap-3 border-b border-[#1b2430] px-5 py-4">
            <span className="icon-box"><Icon name="spark" /></span>
            <div><strong className="block text-sm">FinTwin reasoning agent</strong><span className="block text-[11px] muted">Tools calculate. The model explains.</span></div>
            <div className="ml-auto flex items-center gap-2"><StatusPill tone="good">Grounded</StatusPill><button className={`tab ${showTrace ? "active" : ""}`} onClick={() => setShowTrace((value) => !value)}>Trace {showTrace ? "on" : "off"}</button></div>
          </div>

          <div className="flex max-h-[500px] min-h-[460px] flex-1 flex-col gap-4 overflow-y-auto px-5 py-6" aria-live="polite">
            {!messages.length && (
              <div className="m-auto max-w-md text-center">
                <span className="icon-box mx-auto mb-4 h-12 w-12"><Icon name="command" className="h-6 w-6" /></span>
                <h2 className="m-0 text-xl font-semibold tracking-tight">Ask a decision, not a search query.</h2>
                <p className="mb-0 mt-2 text-sm muted">The copilot routes each question through analytics, risk and simulation tools, then returns the evidence trail.</p>
              </div>
            )}
            {messages.map((message, index) => (
              <article key={index} className={`max-w-[88%] rounded-xl border px-4 py-3 text-sm ${message.role === "user" ? "ml-auto border-[#28534a] bg-[#10221f]" : "border-[#1b2430] bg-[#0b1017]"}`}>
                <div className="mb-2 flex items-center gap-2"><span className="balance-label">{message.role === "user" ? "You" : "FinTwin agent"}</span>{message.meta && <StatusPill tone="violet">{message.meta.intent}</StatusPill>}</div>
                {message.role === "assistant" ? <SafeAnswer text={message.text} /> : <p className="m-0">{message.text}</p>}
                {showTrace && message.meta && (
                  <div className="mt-4 border-t border-[#1b2430] pt-3">
                    <div className="mb-2 flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider muted-2"><Icon name="nodes" className="h-3 w-3" /> Evidence trace · {message.meta.latency_ms.toFixed(0)} ms</div>
                    <div className="flex flex-wrap gap-1.5">
                      {message.meta.tool_calls.map((tool) => <StatusPill key={tool} tone="good">{tool}</StatusPill>)}
                      {message.meta.trace.map((trace, traceIndex) => <StatusPill key={`${trace.node}-${traceIndex}`}>{trace.label ?? trace.node} · {Math.round(trace.ms)}ms</StatusPill>)}
                    </div>
                  </div>
                )}
              </article>
            ))}
            {busy && <div className="max-w-[220px] rounded-xl border border-[#1b2430] bg-[#0b1017] px-4 py-3 text-sm muted"><span className="loading-line"><span />Running financial tools</span></div>}
          </div>

          <div className="border-t border-[#1b2430] p-4">
            <div className="flex gap-2 rounded-xl border border-[#2a3848] bg-[#080c12] p-2 focus-within:border-[#5cf2cc]">
              <input
                ref={inputRef}
                className="min-w-0 flex-1 border-0 bg-transparent px-2 text-sm text-white outline-none placeholder:text-[#56616e]"
                placeholder="Ask about affordability, risk, goals or unusual spending…"
                value={question}
                maxLength={500}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter") ask(question); }}
              />
              <button className="button" onClick={() => ask(question)} disabled={busy || !question.trim()} aria-label="Send question"><Icon name="send" /> Ask</button>
            </div>
            <p className="mb-0 mt-2 text-center text-[10px] muted-2">Every numeric claim comes from a typed FinTwin tool call. Outputs remain probabilistic.</p>
          </div>
        </div>
      </Panel>

      <Panel eyebrow="Quick prompts" title="Decision shortcuts">
        <div className="list-stack">
          {EXAMPLES.map((example) => (
            <button className="list-row text-left" key={example.title} onClick={() => ask(example.text)} disabled={busy}>
              <span className="icon-box"><Icon name={example.icon} /></span>
              <span className="list-row-main"><strong>{example.title}</strong><span>{example.text}</span></span>
              <Icon name="chevron" className="h-4 w-4 muted-2" />
            </button>
          ))}
        </div>
        <div className="action-card mt-5">
          <div className="flex items-center gap-2"><Icon name="lock" className="h-4 w-4 accent" /><strong>Trust boundary</strong></div>
          <p className="mb-0">The language model can select and explain tools. It cannot manufacture calculations or modify your source ledger.</p>
        </div>
      </Panel>
    </div>
  );
}
