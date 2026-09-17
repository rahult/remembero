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
  "Durable memory for AI agents: plain facts and rules, deterministic answers with the proof attached, and a sixty-second live demo — no account, no install, no served model weights. Labs, an in-browser SQLite playground, and measured evidence.";

const designContract = `<!--
THESIS: A first-time visitor with zero context walks problem → idea → try → evidence; the sixty-second live demo proves the mechanism before any vocabulary is needed, labs deepen it, /research carries the measured story, and no trained weights are ever served.
OWN-WORLD: Ledger system — warm paper evidence canvas with a faint graph grid, deep ink chrome, ultramarine execution, amber provenance, green verdicts; Geist controls, mono data, Fraunces display and answers.
STORY: Hero promise with one proof card, then the forgets/misremembers problem, the store-rule-ask idea beside a real in-page engine demo, four workbenches on ink with the IDE showcase, a three-metric evidence teaser, and the models-translate-rules-decide boundary; /research holds the full de-jargonized measurement story.
FIRST VIEWPORT: Editorial hero on graph paper with one proof-carrying answer card, a stamped provenance seal, and the no-weights-served boundary line.
FORM: Progressive editorial product site with a live demo widget, a try section of four workbenches, a separate evidence page at /research, three labs and the SQLite + Datalog IDE at /playground/.
BOUNDARY: The trained reader/writer models are never hosted, served, or required here; claims about them link to measured runs in docs/research/ and use no internal version jargon on visitor surfaces.
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
