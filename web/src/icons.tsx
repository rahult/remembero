import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement> & {
  size?: number;
  strokeWidth?: number;
};

function IconBase({
  children,
  size = 24,
  strokeWidth = 1.9,
  ...props
}: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      width={size}
      height={size}
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

export function AskIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="m16 16 4 4" />
    </IconBase>
  );
}

export function KnowledgeIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M5 5.5A2.5 2.5 0 0 1 7.5 3H19v16H7.5A2.5 2.5 0 0 0 5 21Z" />
      <path d="M5 5.5V21" />
      <path d="M12 7h4" />
    </IconBase>
  );
}

export function GraphIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="6" cy="12" r="2.4" />
      <circle cx="18" cy="7" r="2.4" />
      <circle cx="18" cy="17" r="2.4" />
      <path d="M8.2 10.9 15.6 8" />
      <path d="m8.2 13.1 7.4 2.9" />
    </IconBase>
  );
}

export function RulesIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 4v16" />
      <path d="M6 7h12" />
      <path d="M6 17h12" />
      <path d="m8.5 7-2.5 4h5Z" />
      <path d="m15.5 17-2.5-4h5Z" />
    </IconBase>
  );
}

export function PlusIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </IconBase>
  );
}

export function SparkIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m12 3 1.6 5.1L19 10l-5.4 1.9L12 17l-1.6-5.1L5 10l5.4-1.9Z" />
    </IconBase>
  );
}

export function GearIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="3.25" />
      <path d="M12 2.75v2.1" />
      <path d="M12 19.15v2.1" />
      <path d="m4.93 4.93 1.48 1.48" />
      <path d="m17.59 17.59 1.48 1.48" />
      <path d="M2.75 12h2.1" />
      <path d="M19.15 12h2.1" />
      <path d="m4.93 19.07 1.48-1.48" />
      <path d="m17.59 6.41 1.48-1.48" />
    </IconBase>
  );
}

export function FolderIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M3.5 6.5A2.5 2.5 0 0 1 6 4h4l2 2h6A2.5 2.5 0 0 1 20.5 8.5v8A2.5 2.5 0 0 1 18 19H6a2.5 2.5 0 0 1-2.5-2.5Z" />
    </IconBase>
  );
}

export function MenuIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M4 7h16" />
      <path d="M4 12h16" />
      <path d="M4 17h16" />
    </IconBase>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m6 6 12 12" />
      <path d="M18 6 6 18" />
    </IconBase>
  );
}

export function SendIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M20 4 9 15" />
      <path d="m20 4-7 16-2.5-7.5L3 10Z" />
    </IconBase>
  );
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m6 9 6 6 6-6" />
    </IconBase>
  );
}

export function ChevronRightIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m9 6 6 6-6 6" />
    </IconBase>
  );
}

export function PulseIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M3 12h4l2.2-5 3.3 10 2.3-5H21" />
    </IconBase>
  );
}

export function DatabaseIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <ellipse cx="12" cy="6" rx="6.5" ry="2.8" />
      <path d="M5.5 6v5.5c0 1.55 2.9 2.8 6.5 2.8s6.5-1.25 6.5-2.8V6" />
      <path d="M5.5 11.5V17c0 1.55 2.9 2.8 6.5 2.8s6.5-1.25 6.5-2.8v-5.5" />
    </IconBase>
  );
}

export function ShieldIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 3.5 18.5 6v5.4c0 4.2-2.3 6.8-6.5 9.1-4.2-2.3-6.5-4.9-6.5-9.1V6Z" />
    </IconBase>
  );
}

export function CalendarIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M6 4.5v3" />
      <path d="M18 4.5v3" />
      <rect x="4" y="6.5" width="16" height="13" rx="2.25" />
      <path d="M4 10.5h16" />
    </IconBase>
  );
}

export function DocumentIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M7 3.5h7l4 4v13H7a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2Z" />
      <path d="M14 3.5v4h4" />
      <path d="M9 12h6" />
      <path d="M9 16h4" />
    </IconBase>
  );
}

export function DocumentEvidenceIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M5.5 4.5h9l4 4v11a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2Z" />
      <path d="M14.5 4.5v4h4" />
      <path d="M8.5 11h7" />
      <path d="M8.5 15h5" />
      <circle cx="7" cy="18" r="1.2" fill="currentColor" stroke="none" />
    </IconBase>
  );
}

export function ProofIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M6 5.5h9l3 3v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-11a2 2 0 0 1 2-2Z" />
      <path d="M15 5.5v3h3" />
      <path d="m8.5 14 2.2 2.2 4.8-5.1" />
    </IconBase>
  );
}

export function RefreshIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M18.5 8A7 7 0 1 0 19 14" />
      <path d="M18.5 3.5V8H14" />
    </IconBase>
  );
}

export function CheckCircleIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="8.25" />
      <path d="m8.7 12.2 2.1 2.1 4.5-4.9" />
    </IconBase>
  );
}

export function SearchLineIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="10.5" cy="10.5" r="5.5" />
      <path d="m15.25 15.25 3.75 3.75" />
    </IconBase>
  );
}

export function ArrowUpRightIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M7 17 17 7" />
      <path d="M9 7h8v8" />
    </IconBase>
  );
}

export function EllipsisIcon(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="6" cy="12" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="18" cy="12" r="1.2" fill="currentColor" stroke="none" />
    </IconBase>
  );
}
