import type { Metadata } from "next";
import { ChatMemoryLab } from "./chat-memory-lab";

export const metadata: Metadata = {
  title: "Remembero Lab — Same database, different powers",
  description:
    "Four questions where SQL and Remembero structurally diverge over one shared browser-local SQLite database: recursive proof chains, refused contradictions, proven absences, and why-not diagnoses.",
};

export default function ChatMemoryPage() {
  return <ChatMemoryLab />;
}
