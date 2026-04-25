/**
 * PLAN-mode discovery context: logical steps (one per tool call) and a pure
 * renderer to AgentMessage[] so duplicate discovery calls can drop older spans
 * without breaking tool_call_id pairing.
 */

import type { AssistantMessage, ToolResultMessage } from "@mariozechner/pi-ai";
import type { AgentMessage, AgentToolCall } from "./types.js";

/** Discovery tools only — `plan` is intentionally excluded from dedupe. */
export const PLAN_DISCOVERY_TOOL_NAMES = ["bash", "find", "grep", "ls", "read"] as const;
export type PlanDiscoveryToolName = (typeof PLAN_DISCOVERY_TOOL_NAMES)[number];

const DISCOVERY_SET = new Set<string>(PLAN_DISCOVERY_TOOL_NAMES);

export function isPlanDiscoveryToolName(name: string): boolean {
	return DISCOVERY_SET.has(name);
}

function normalizePathLike(v: unknown): string {
	if (typeof v !== "string") return "";
	let s = v.trim().replace(/\\/g, "/");
	if (s.startsWith("./")) s = s.slice(2);
	if (s.startsWith("@/")) s = s.slice(2);
	return s;
}

function collapseWhitespace(s: string): string {
	return s.trim().replace(/\s+/g, " ");
}

/** Normalize tool arguments for stable dedupe keys (per-tool tweaks). */
export function normalizeDiscoveryToolArguments(toolName: string, raw: unknown): unknown {
	if (raw === null || raw === undefined) return null;
	if (typeof raw !== "object" || Array.isArray(raw)) return { _v: raw };
	const o = { ...(raw as Record<string, unknown>) };
	switch (toolName) {
		case "read": {
			if (typeof o.path === "string") o.path = normalizePathLike(o.path);
			break;
		}
		case "grep":
		case "find":
		case "ls": {
			if (typeof o.path === "string") o.path = normalizePathLike(o.path) || ".";
			if (typeof o.pattern === "string") o.pattern = o.pattern.trim();
			if (typeof o.glob === "string") o.glob = o.glob.trim();
			break;
		}
		case "bash": {
			if (typeof o.command === "string") o.command = collapseWhitespace(o.command);
			if (typeof o.cwd === "string") o.cwd = normalizePathLike(o.cwd);
			break;
		}
		default:
			break;
	}
	return sortKeysDeep(o);
}

function sortKeysDeep(v: unknown): unknown {
	if (v === null || v === undefined) return v;
	if (typeof v !== "object") return v;
	if (Array.isArray(v)) return v.map(sortKeysDeep);
	const obj = v as Record<string, unknown>;
	const out: Record<string, unknown> = {};
	for (const k of Object.keys(obj).sort()) {
		out[k] = sortKeysDeep(obj[k]);
	}
	return out;
}

export function stableStringifyForDedupe(v: unknown): string {
	try {
		return JSON.stringify(v);
	} catch {
		return String(v);
	}
}

export function canonicalDiscoveryDedupeKey(toolName: string, rawArgs: unknown): string {
	const norm = normalizeDiscoveryToolArguments(toolName, rawArgs);
	return `${toolName}\u0000${stableStringifyForDedupe(norm)}`;
}

export type PlanDiscoveryStep = {
	toolCallId: string;
	toolName: string;
	normalizedArgs: unknown;
	rawArgs: unknown;
	result: ToolResultMessage;
	assistantPreamble?: string;
	assistantApi: AssistantMessage["api"];
	assistantProvider: AssistantMessage["provider"];
	assistantModel: AssistantMessage["model"];
	assistantUsage: AssistantMessage["usage"];
	canonicalKey: string;
};

export function extractAssistantPreambleText(msg: AssistantMessage): string | undefined {
	const parts = msg.content
		.filter((c): c is { type: "text"; text: string } => c.type === "text" && typeof (c as { text?: unknown }).text === "string")
		.map((c) => c.text.trim())
		.filter((t) => t.length > 0);
	if (parts.length === 0) return undefined;
	return parts.join("\n\n");
}

export function cloneToolResultMessage(tr: ToolResultMessage): ToolResultMessage {
	return {
		...tr,
		content: Array.isArray(tr.content) ? tr.content.map((c) => ({ ...c })) : tr.content,
		details: tr.details,
	};
}

/** One assistant (single toolCall) + matching toolResult per step — valid pairing. */
export function stepsToPlanDiscoveryMessages(steps: PlanDiscoveryStep[]): AgentMessage[] {
	const out: AgentMessage[] = [];
	for (const s of steps) {
		const preamble = s.assistantPreamble?.trim();
		const content: AssistantMessage["content"] = [];
		if (preamble) {
			content.push({ type: "text", text: preamble });
		}
		content.push({
			type: "toolCall",
			id: s.toolCallId,
			name: s.toolName,
			arguments: s.rawArgs as Record<string, unknown>,
		});
		const assistant: AssistantMessage = {
			role: "assistant",
			content,
			api: s.assistantApi,
			provider: s.assistantProvider,
			model: s.assistantModel,
			usage: s.assistantUsage,
			stopReason: "toolUse",
			timestamp: s.result.timestamp,
		};
		out.push(assistant);
		out.push(s.result);
	}
	return out;
}

export function buildPlanDiscoveryStepsFromToolBatch(
	assistant: AssistantMessage,
	toolCalls: AgentToolCall[],
	toolResults: ToolResultMessage[],
): PlanDiscoveryStep[] {
	const preamble = extractAssistantPreambleText(assistant);
	const out: PlanDiscoveryStep[] = [];
	for (let i = 0; i < toolCalls.length; i++) {
		const tc = toolCalls[i];
		const tr = toolResults[i];
		if (!tc || tc.type !== "toolCall") continue;
		if (!isPlanDiscoveryToolName(tc.name)) continue;
		out.push({
			toolCallId: tc.id,
			toolName: tc.name,
			normalizedArgs: normalizeDiscoveryToolArguments(tc.name, tc.arguments),
			rawArgs: tc.arguments,
			result: cloneToolResultMessage(tr),
			assistantPreamble: preamble,
			assistantApi: assistant.api,
			assistantProvider: assistant.provider,
			assistantModel: assistant.model,
			assistantUsage: assistant.usage,
			canonicalKey: canonicalDiscoveryDedupeKey(tc.name, tc.arguments),
		});
	}
	return out;
}

/**
 * Drop any prior step with the same canonicalKey (in order), then append each
 * incoming step in batch order so chronology of non-duplicates is preserved.
 */
export function mergePlanDiscoverySteps(
	existing: PlanDiscoveryStep[],
	incomingBatch: PlanDiscoveryStep[],
): { steps: PlanDiscoveryStep[]; replacedKeys: string[] } {
	let steps = existing.slice();
	const replacedKeys: string[] = [];
	for (const b of incomingBatch) {
		const had = steps.some((s) => s.canonicalKey === b.canonicalKey);
		if (had) replacedKeys.push(b.canonicalKey);
		steps = steps.filter((s) => s.canonicalKey !== b.canonicalKey);
		steps.push(b);
	}
	return { steps, replacedKeys };
}
