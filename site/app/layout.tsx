import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
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

const title = "Remembero — Memory you can reason with";
const description =
  "Proof-carrying memory for agents that runs on its own 2.3B model: readable facts, deterministic rules, computed notes, real-life browser labs, an SQLite-native playground, and worked examples with their numbers attached.";

const designContract = `<!--
THESIS: The main site sells proof-carrying memory; two labs show the real-life effect; the playground exposes the mechanism.
OWN-WORLD: True white evidence canvas, navy structural chrome, cobalt execution, amber provenance, compact sans controls, mono data, serif answers.
STORY: A visitor understands the product, compares real-life model behavior in /labs/, then opens /playground/ to inspect the machinery.
FIRST VIEWPORT: Marketing hero with one proof-carrying answer, product promise, and direct Playground and GitHub actions.
FORM: Editorial product site plus two comparison labs and a separate Guided Query Canvas at /playground/.
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
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
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
