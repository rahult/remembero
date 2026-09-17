import type { Metadata } from "next";
import { Fraunces, Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  style: ["normal", "italic"],
});

const title = "Remembero — Memory you can reason with";
const description =
  "Proof-carrying memory for agents: readable facts, deterministic rules, computed notes, and a reading-recall research trail with every number paired-run measured. Browser-local playground and labs — no model weights served.";

const designContract = `<!--
THESIS: The main site sells proof-carrying memory and shows the research arc that built it; three labs and the playground demonstrate the mechanism with zero served weights.
OWN-WORLD: Ledger system — warm paper evidence canvas with a faint graph grid, deep ink chrome, ultramarine execution, amber provenance, green verdicts; Geist controls, mono data, Fraunces display and answers.
STORY: A visitor understands the product, reads the paired-run research ledger, then works the labs and playground — all deterministic in-browser, with model output only as labeled replays or optional third-party WebLLM.
FIRST VIEWPORT: Editorial hero on graph paper with one proof-carrying answer card, a stamped provenance seal, and the no-weights-served boundary line.
FORM: Editorial product site, a research delta ledger, three labs (chat recall, grounded agent, reading recall) and the SQLite + Datalog IDE at /playground/.
BOUNDARY: The latest trained reader/writer models are never hosted, served, or required here; claims about them link to the measured runs in docs/research/.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
-->`;

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("host") ?? "localhost:3000";
  const forwardedProtocol = requestHeaders.get("x-forwarded-proto");
  const protocol = forwardedProtocol ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const image = `${origin}/og.png`;
  return {
    metadataBase: new URL(origin),
    title,
    description,
    openGraph: {
      type: "website",
      url: origin,
      title,
      description,
      images: [{ url: image, width: 1731, height: 909, alt: title }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [image],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${fraunces.variable} antialiased`}
      >
        <template
          id="rembero-design-contract"
          dangerouslySetInnerHTML={{ __html: designContract }}
        />
        {children}
      </body>
    </html>
  );
}
