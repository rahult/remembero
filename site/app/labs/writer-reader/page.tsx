import type { Metadata } from "next";
import { WriterReaderLab } from "./writer-reader-lab";

export const metadata: Metadata = {
  title: "Writer–Reader Lab — Raw text in, proven answers out",
  description:
    "Six months of genuinely messy chat — corrections, a flip-back at a new price, an effective-dated handover, a contract that ends, and one question never answered anywhere. Watch the writer turn text into claims, watch code build validity timelines, then ask the questions that punish naive retrieval. No model weights served.",
};

export default function WriterReaderPage() {
  return <WriterReaderLab />;
}
