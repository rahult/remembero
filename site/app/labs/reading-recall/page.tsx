import type { Metadata } from "next";
import { ReadingRecallLab } from "./reading-recall-lab";

export const metadata: Metadata = {
  title: "Reading Recall Lab — The reader contract, model taken out",
  description:
    "The reading-recall pipeline running deterministically in your browser: lexical retrieval, a re-rank stand-in, context tiering, and computed notes written live — then both arms of a recorded paired run, the same reader with and without the notes. No model weights served.",
};

export default function ReadingRecallPage() {
  return <ReadingRecallLab />;
}
