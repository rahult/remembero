"use client";

import { useMemo } from "react";
import type { BrowserDatalogProof } from "../lib/sqlite-wasm";

interface ProofGraphProps {
  proof: BrowserDatalogProof | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

interface TreeNode {
  id: string;
  proof: BrowserDatalogProof;
  children: TreeNode[];
}

interface PositionedNode extends TreeNode {
  x: number;
  y: number;
  depth: number;
}

interface Edge {
  from: string;
  to: string;
}

const MIN_WIDTH = 360;
const NODE_WIDTH = 140;
const NODE_GAP = 34;
const SIDE_PADDING = 18;
const LEVEL_HEIGHT = 112;

function treeFor(proof: BrowserDatalogProof, id = "proof"): TreeNode {
  return {
    id,
    proof,
    children: (proof.because ?? []).map((child, index) =>
      treeFor(child, `${id}.${index}`),
    ),
  };
}

function countLeaves(node: TreeNode): number {
  if (node.children.length === 0) return 1;
  return node.children.reduce((total, child) => total + countLeaves(child), 0);
}

function layout(root: TreeNode): {
  nodes: PositionedNode[];
  edges: Edge[];
  width: number;
  height: number;
} {
  const nodes: PositionedNode[] = [];
  const edges: Edge[] = [];
  const leaves = countLeaves(root);
  const width = Math.max(
    MIN_WIDTH,
    SIDE_PADDING * 2 + leaves * NODE_WIDTH + Math.max(0, leaves - 1) * NODE_GAP,
  );
  let leafIndex = 0;
  let maxDepth = 0;

  const position = (node: TreeNode, depth: number): number => {
    maxDepth = Math.max(maxDepth, depth);
    const childPositions = node.children.map((child) => {
      edges.push({ from: node.id, to: child.id });
      return position(child, depth + 1);
    });
    const x =
      childPositions.length === 0
        ? ((leafIndex++ + 0.5) / leaves) * width
        : childPositions.reduce((sum, value) => sum + value, 0) /
          childPositions.length;
    nodes.push({ ...node, x, y: 42 + depth * LEVEL_HEIGHT, depth });
    return x;
  };

  position(root, 0);
  return { nodes, edges, width, height: 92 + maxDepth * LEVEL_HEIGHT };
}

function nodeLabel(proof: BrowserDatalogProof): { title: string; detail: string } {
  return {
    title: proof.predicate,
    detail: `(${proof.values.join(", ")})`,
  };
}

export function ProofGraph({ proof, selectedId, onSelect }: ProofGraphProps) {
  const graph = useMemo(() => (proof ? layout(treeFor(proof)) : null), [proof]);

  if (!graph) {
    return (
      <div className="ide-empty compact">
        <strong>No proof graph yet</strong>
        <span>Run a prepared rule to project its support.</span>
      </div>
    );
  }

  const byId = new Map(graph.nodes.map((node) => [node.id, node]));

  return (
    <div className="proof-graph-wrap">
      <svg
        className="proof-graph"
        width={graph.width}
        height={graph.height}
        viewBox={`0 0 ${graph.width} ${graph.height}`}
        role="img"
        aria-labelledby="proof-graph-title proof-graph-description"
      >
        <title id="proof-graph-title">Query-scoped proof graph</title>
        <desc id="proof-graph-description">
          The derived answer is connected to every SQLite fact used by its rule.
        </desc>
        {graph.edges.map((edge) => {
          const from = byId.get(edge.from)!;
          const to = byId.get(edge.to)!;
          return (
            <path
              key={`${edge.from}-${edge.to}`}
              className="proof-edge"
              d={`M ${from.x} ${from.y + 25} C ${from.x} ${from.y + 64}, ${to.x} ${to.y - 64}, ${to.x} ${to.y - 25}`}
            />
          );
        })}
        {graph.nodes.map((node) => {
          const label = nodeLabel(node.proof);
          const selected = node.id === selectedId;
          const derived = node.proof.rule !== undefined;
          return (
            <g
              key={node.id}
              className={`proof-node ${derived ? "derived" : "fact"}${selected ? " selected" : ""}`}
              role="button"
              tabIndex={0}
              aria-label={`${derived ? "Derived" : "SQLite fact"}: ${label.title} ${label.detail}`}
              onClick={() => onSelect(node.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(node.id);
                }
              }}
              transform={`translate(${node.x}, ${node.y})`}
            >
              <rect x={-NODE_WIDTH / 2} y="-25" width={NODE_WIDTH} height="50" rx="9" />
              <text textAnchor="middle" y="-3">
                {label.title.length > 18 ? `${label.title.slice(0, 17)}…` : label.title}
              </text>
              <text className="proof-node-detail" textAnchor="middle" y="14">
                {label.detail.length > 24 ? `${label.detail.slice(0, 23)}…` : label.detail}
              </text>
            </g>
          );
        })}
      </svg>

      <details className="graph-list">
        <summary>Graph as an ordered list</summary>
        <ol>
          {graph.nodes
            .slice()
            .sort((left, right) => left.depth - right.depth || left.x - right.x)
            .map((node) => {
              const label = nodeLabel(node.proof);
              return (
                <li key={node.id}>
                  <button
                    type="button"
                    aria-pressed={node.id === selectedId}
                    onClick={() => onSelect(node.id)}
                  >
                    <strong>{label.title}</strong>
                    <code>{label.detail}</code>
                  </button>
                </li>
              );
            })}
        </ol>
      </details>
    </div>
  );
}
