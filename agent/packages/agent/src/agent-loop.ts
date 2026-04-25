/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type Context,
	EventStream,
	streamSimple,
	type ToolResultMessage,
	validateToolArguments,
} from "@mariozechner/pi-ai";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	StreamFn,
} from "./types.js";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import type { Dirent } from "node:fs";
import { basename, extname, isAbsolute, join, relative, resolve, sep } from "node:path";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/** Cap echoed JSON in the plan draft handshake so one pathological payload cannot blow the context window. */
const PLAN_DRAFT_ECHO_JSON_MAX_CHARS = 100_000;

/** True if `candidate` is the workspace root or a path strictly inside it (handles Windows drives). */
function isPathInsideWorkspaceRoot(root: string, candidate: string): boolean {
	let rootAbs = resolve(root);
	let candAbs = resolve(candidate);
	if (process.platform === "win32") {
		rootAbs = rootAbs.toLowerCase();
		candAbs = candAbs.toLowerCase();
	}
	if (candAbs === rootAbs) return true;
	const prefix = rootAbs.endsWith(sep) ? rootAbs : rootAbs + sep;
	return candAbs.startsWith(prefix);
}

function safeResolvePathUnderCwd(requestedPath: string): { abs: string; ok: boolean } {
	const cwd = process.cwd();
	const abs = resolve(cwd, requestedPath);
	return { abs, ok: isPathInsideWorkspaceRoot(cwd, abs) };
}

const UNICODE_SPACE_CHARS = /[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g;

/** Normalize model-supplied paths (quotes, file URLs, backslashes, unicode spaces, accidental aliases). */
function normalizeToolPathString(raw: string): string {
	let s = raw.trim().replace(UNICODE_SPACE_CHARS, " ");
	if (
		(s.startsWith('"') && s.endsWith('"')) ||
		(s.startsWith("'") && s.endsWith("'")) ||
		(s.startsWith("`") && s.endsWith("`"))
	) {
		s = s.slice(1, -1).trim().replace(UNICODE_SPACE_CHARS, " ");
	}
	if (s.startsWith("file:")) {
		try {
			const u = new URL(s);
			let p = u.pathname;
			if (process.platform === "win32" && p.startsWith("/") && /^\/[A-Za-z]:/.test(p)) {
				p = p.slice(1);
			}
			s = decodeURIComponent(p);
		} catch {
			/* keep s */
		}
	}
	s = s.replace(/\\/g, "/");
	if (s.startsWith("@/")) {
		s = s.slice(2);
	}
	if (s.startsWith("./")) {
		s = s.slice(2);
	}
	if (s.length === 0) {
		return ".";
	}
	return s;
}

function toRepoRelativePosixPath(cwd: string, absolutePath: string): string {
	let rel = relative(cwd, absolutePath);
	if (!rel || rel === "") return ".";
	if (rel.startsWith("..") || isAbsolute(rel)) {
		return absolutePath.split(sep).join("/");
	}
	return rel.split(sep).join("/");
}

const BASENAME_WALK_SKIP_DIRS = new Set([
	"node_modules",
	".git",
	"dist",
	".next",
	"build",
	"coverage",
	"__pycache__",
	".turbo",
	".cache",
	".venv",
	"venv",
]);

/**
 * Walk the workspace (cwd) for files whose basename exactly matches `exactBasename`.
 * Returns repo-relative POSIX paths, excluding heavy directories.
 */
function findRepoFilesByExactBasename(cwd: string, exactBasename: string, maxMatches: number): string[] {
	const results: string[] = [];
	const queue: string[] = [cwd];
	let scanned = 0;
	const MAX_NODES = 100_000;
	while (queue.length > 0 && results.length < maxMatches && scanned < MAX_NODES) {
		const dir = queue.pop()!;
		let entries: Dirent[];
		try {
			entries = readdirSync(dir, { withFileTypes: true });
		} catch {
			continue;
		}
		for (const ent of entries) {
			scanned++;
			if (ent.isSymbolicLink()) {
				continue;
			}
			const full = join(dir, ent.name);
			if (ent.isDirectory()) {
				if (BASENAME_WALK_SKIP_DIRS.has(ent.name)) continue;
				queue.push(full);
			} else if (ent.isFile() && ent.name === exactBasename) {
				results.push(toRepoRelativePosixPath(cwd, full));
			}
		}
	}
	return results;
}

function rankBasenameMatches(originalNorm: string, matches: string[]): string {
	if (matches.length === 1) return matches[0];
	const needle = originalNorm.replace(/\\/g, "/").replace(/^\.?\//, "").replace(/\/$/, "");
	const segments = needle.split("/").filter(Boolean);
	const scorePath = (p: string): number => {
		const pn = p.replace(/\\/g, "/").replace(/^\.\//, "");
		let s = 0;
		if (pn === needle || pn.endsWith("/" + needle) || needle.endsWith("/" + pn) || pn.endsWith(needle)) {
			s += 10_000;
		}
		for (let i = 1; i <= Math.min(segments.length, 8); i++) {
			const suff = segments.slice(-i).join("/");
			if (suff.length > 0 && pn.endsWith(suff)) s += i * 100;
		}
		return s;
	};
	const ranked = [...matches].sort((a, b) => {
		const d = scorePath(b) - scorePath(a);
		if (d !== 0) return d;
		return a.localeCompare(b);
	});
	return ranked[0]!;
}

type ResolveWorkspacePathOptions = {
	/** When the path does not exist, search the repo for a file with the same basename (read/edit only). */
	basenameFallback: boolean;
};

/**
 * Resolve a single path argument to a safe, repo-relative POSIX path when possible.
 * Applies to all file/directory tool args in `applyRobustWorkspacePathsToToolCall`.
 */
function resolveWorkspaceToolPathString(raw: string, opts: ResolveWorkspacePathOptions): string {
	const cwd = process.cwd();
	const norm = normalizeToolPathString(raw);
	const { abs, ok } = safeResolvePathUnderCwd(norm);
	if (!ok) {
		return norm;
	}
	try {
		if (existsSync(abs)) {
			const st = statSync(abs);
			if (st.isFile() || st.isDirectory()) {
				return toRepoRelativePosixPath(cwd, abs);
			}
		}
	} catch {
		return norm;
	}

	if (!opts.basenameFallback) {
		return norm;
	}

	const base = basename(norm.replace(/\/$/, ""));
	if (base.length === 0 || base === "." || base === "..") {
		return norm;
	}

	const matches = findRepoFilesByExactBasename(cwd, base, 32);
	if (matches.length === 0) {
		return norm;
	}

	return rankBasenameMatches(norm, matches);
}

function rawPathFromToolArguments(args: unknown): string | undefined {
	if (!args || typeof args !== "object" || Array.isArray(args)) return undefined;
	const r = args as Record<string, unknown>;
	if (typeof r.path === "string" && r.path.trim().length > 0) return r.path;
	if (typeof r.file_path === "string" && r.file_path.trim().length > 0) return r.file_path;
	return undefined;
}

/** Match `prepareToolCall` path logic so loop state (edit failures, reads) tracks the same path the tool executed. */
function resolvedLoopPathForTool(tc: AgentToolCall): string | undefined {
	const raw = rawPathFromToolArguments(tc.arguments);
	if (!raw) return undefined;
	if (tc.name === "edit" || tc.name === "read") {
		return resolveWorkspaceToolPathString(raw, { basenameFallback: true });
	}
	if (tc.name === "write") {
		return resolveWorkspaceToolPathString(raw, { basenameFallback: false });
	}
	return raw;
}

function applyRobustWorkspacePathsToToolCall(toolCall: AgentToolCall): AgentToolCall {
	const args = toolCall.arguments;
	if (!args || typeof args !== "object" || Array.isArray(args)) {
		return toolCall;
	}
	const rec = args as Record<string, unknown>;
	const name = toolCall.name;
	const next: Record<string, unknown> = { ...rec };
	let changed = false;

	const touchPathKey = (key: "path" | "file_path", basenameFallback: boolean) => {
		if (typeof next[key] !== "string") return;
		const raw = next[key] as string;
		const resolved = resolveWorkspaceToolPathString(raw, { basenameFallback });
		if (resolved !== raw) {
			next[key] = resolved;
			changed = true;
		}
	};

	if (name === "read" || name === "edit") {
		touchPathKey("path", true);
		touchPathKey("file_path", true);
	} else if (name === "write") {
		touchPathKey("path", false);
		touchPathKey("file_path", false);
	} else if (name === "grep" || name === "find" || name === "ls") {
		touchPathKey("path", false);
	}

	return changed ? { ...toolCall, arguments: next as Record<string, any> } : toolCall;
}

/** Track reads under aliases the model may reuse (`./a` vs `a`) for nudges. */
function addReadPathVariants(set: Set<string>, readPath: string): void {
	set.add(readPath);
	const isAbs = readPath.startsWith("/") || /^[A-Za-z]:[\\/]/.test(readPath);
	if (isAbs) return;
	const stripped = readPath.replace(/^\.\//, "");
	set.add(stripped);
	if (!readPath.startsWith("./")) set.add(`./${stripped}`);
}

/** Avoid breaking Markdown fences if the model or file contains triple backticks. */
function escapeMarkdownFences(s: string): string {
	return s.replace(/```/g, "\\`\\`\\`");
}

const GEMINI_OLDTEXT_RECOVERY_MAX_CHARS = 500_000;

/**
 * Summarize each edits[] entry as the model sent it, including its lineRange (if any)
 * and its oldText. Used by the recovery message so the model can see exactly what it
 * asked for versus what the file actually contains at those lines.
 */
function extractEditsForRecovery(toolCall: AgentToolCall): string {
	const args = toolCall.arguments as
		| {
				edits?: Array<Record<string, unknown>>;
				oldText?: unknown;
				startLine?: unknown;
				endLine?: unknown;
				lineRange?: unknown;
		  }
		| undefined;
	if (!args) return "";
	const chunks: string[] = [];
	const fmtRange = (r: Record<string, unknown> | undefined): string => {
		if (!r) return "(missing or invalid)";
		const pairs: Array<[unknown, unknown]> = [
			[r.startLine, r.endLine],
			[r.start_line, r.end_line],
			[r.firstLine, r.lastLine],
			[r.first_line, r.last_line],
		];
		for (const [a, b] of pairs) {
			if (typeof a === "number" && typeof b === "number") return `[${a}, ${b}]`;
		}
		const t = r.lineRange;
		if (Array.isArray(t) && t.length === 2 && typeof t[0] === "number" && typeof t[1] === "number") {
			return `[${t[0]}, ${t[1]}]`;
		}
		return "(missing or invalid)";
	};
	if (typeof args.oldText === "string" && args.oldText.length > 0) {
		chunks.push(`range=${fmtRange(args as Record<string, unknown>)}\noldText:\n${args.oldText}`);
	}
	if (Array.isArray(args.edits)) {
		for (let i = 0; i < args.edits.length; i++) {
			const e = args.edits[i] ?? {};
			const ot = typeof e?.oldText === "string" ? (e.oldText as string) : "";
			const range = fmtRange(e as Record<string, unknown>);
			chunks.push(`edits[${i}]: range=${range}\noldText:\n${ot}`);
		}
	}
	let s = chunks.join("\n---\n");
	if (s.length > GEMINI_OLDTEXT_RECOVERY_MAX_CHARS) {
		s = `${s.slice(0, GEMINI_OLDTEXT_RECOVERY_MAX_CHARS)}…`;
	}
	return s;
}

/**
 * Prepend 0-indexed line numbers to the file content (right-padded to 6 chars).
 * Matches the indexing convention of the line-range edit tool so the model can
 * pick `lineRange: [first, last]` directly off the shown numbering.
 */
function formatFileWithZeroIndexedLines(content: string): string {
	const lines = content.split("\n");
	const pad = Math.max(3, String(Math.max(0, lines.length - 1)).length);
	const out: string[] = [];
	for (let i = 0; i < lines.length; i++) {
		const n = String(i).padStart(pad, " ");
		out.push(`${n}| ${lines[i]}`);
	}
	return out.join("\n");
}

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<void> {
	let firstTurn = true;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	let upstreamRetries = 0;
	const UPSTREAM_RETRY_LIMIT = 100;

	const editFailMap = new Map<string, number>();
	const editNotFoundStreakMap = new Map<string, number>();
	const priorFailedAnchor = new Map<string, string>();

	let explorationCount = 0;
	let totalExplorationSteps = 0;
	let hasProducedEdit = false;

	const loopStart = Date.now();
	const pathsAlreadyRead = new Set<string>();
	const pathReadCounts = new Map<string, number>();
	let lastRereadNudgeAt = 0;
	const editedPaths = new Set<string>();
	const pathEditCounts = new Map<string, number>();
	let consecutiveEditsOnSameFile = 0;
	let lastEditedFile = "";

	let executionMode: "plan" | "implement" = "plan";
	let planSubmitted = false;
	/**
	 * Plan handshake: first **plan-only** turn echoes the payload without running
	 * repository/criteria validation. Any turn that includes a non-`plan` tool
	 * resets this flag. A later **plan-only** turn (with this flag true) runs
	 * the real `plan` tool and, if it passes, transitions to IMPLEMENT.
	 */
	let planHandshakeAwaitingConsecutivePlanOnly = false;
	/**
	 * Ultimate wall-clock time limit for the whole run (measured from loopStart).
	 * When this is hit we end the run *successfully* even if not every plan is
	 * finished. It exists so we always return a partial patch before the outer
	 * harness (Docker 300s) hard-kills the process.
	 */
	const ULTIMATE_TIME_LIMIT_MS = 280_000;
	/**
	 * Minimum per-plan budget. Even when we are very late in the run, each plan
	 * gets at least this many ms before the auto-advance kicks in so the model
	 * has a real chance to land a single edit.
	 */
	const MIN_PLAN_BUDGET_MS = 15_000;
	/**
	 * We warn the model about the time-budget when less than this much remains
	 * inside the current plan's budget.
	 */
	const PLAN_BUDGET_WARN_MS = 20_000;
	const pendingTurnEndMarkerLines: string[] = [];
	const rawEmit = emit;
	emit = async (event: AgentEvent) => {
		if (event.type !== "turn_end" || pendingTurnEndMarkerLines.length === 0) {
			return rawEmit(event);
		}
		const markerBlock = pendingTurnEndMarkerLines.join("\n");
		pendingTurnEndMarkerLines.length = 0;
		const m = event.message;
		if (m.role !== "assistant" || !Array.isArray(m.content)) {
			return rawEmit(event);
		}
		const markerText = `[agent_loop_event]\n${markerBlock}`;
		const nextContent = [...m.content, { type: "text" as const, text: markerText }];
		const nextEvent: AgentEvent = {
			...event,
			message: {
				...m,
				content: nextContent,
			},
		};
		return rawEmit(nextEvent);
	};
	let foundFiles: string[] = [];
	let absorbedFiles = new Set<string>();
	const plannedFiles = new Set<string>();
	const planByFile = new Map<string, string>();

	/**
	 * Implement mode is a simple sequential for-loop over plans.
	 * `plannedOrder` preserves the submission order from the `plan` tool call;
	 * `currentPlanIndex` is the plan currently being worked on. We advance past a
	 * plan when the model calls the `editdone` tool. When `currentPlanIndex`
	 * reaches `plannedOrder.length`, the run ends.
	 */
	const plannedOrder: Array<{ path: string; plan: string }> = [];
	let currentPlanIndex = 0;
	/** Per-plan snapshot of file content at the moment that plan starts. */
	const originalContentByPlanIndex = new Map<number, string>();
	/**
	 * Handshake state for editdone:
	 * first editdone -> send confirmation inject;
	 * second consecutive detailed editdone on same plan -> advance.
	 */
	let pendingEditdoneConfirmationPlanIndex: number | null = null;
	/**
	 * Timestamp (ms) when the current plan started being worked on. Reset every
	 * time `currentPlanIndex` advances (on `editdone` or budget-auto-advance) and
	 * initialised when implement mode starts. Used together with the per-plan
	 * budget to (a) tell the model how much time it has, and (b) auto-advance if
	 * the plan runs over budget.
	 */
	let currentPlanStartedAt: number | null = null;

	/**
	 * Compute per-plan budget from (remaining time until ULTIMATE_TIME_LIMIT_MS)
	 * divided evenly across the remaining plans. Floored to MIN_PLAN_BUDGET_MS
	 * so we always give every plan a real shot even when time is tight.
	 */
	const computePlanBudgetMs = (): number => {
		const elapsed = Date.now() - loopStart;
		const remainingToUltimate = Math.max(0, ULTIMATE_TIME_LIMIT_MS - elapsed);
		const remainingPlans = Math.max(1, plannedOrder.length - currentPlanIndex);
		const share = Math.floor(remainingToUltimate / remainingPlans);
		return Math.max(MIN_PLAN_BUDGET_MS, share);
	};
	/**
	 * Paths the model successfully `read` during PLAN mode. Surfaced to the model
	 * during IMPLEMENT mode so it knows which surrounding files are already in
	 * scope and can `read` them again for additional context before editing.
	 * Insertion-order is preserved.
	 */
	const planModeReadPaths: string[] = [];
	const planModeReadPathSet = new Set<string>();
	const recordPlanModeReadPath = (path: string): void => {
		if (!path) return;
		const norm = normalizePathForMatch(path);
		if (!norm || planModeReadPathSet.has(norm)) return;
		planModeReadPathSet.add(norm);
		planModeReadPaths.push(norm);
	};

	/**
	 * Build the implement-mode injection message for the current plan.
	 * Contains: time-budget block, plan text, full current content of the target
	 * file with 0-indexed line numbers, and an instruction to call
	 * edit / write / editdone.
	 *
	 * `planStartedAt` is the wall-clock timestamp when the current plan started
	 * being worked on; `budgetMs` is the per-plan budget computed for this plan.
	 * Both are surfaced to the model so it understands its time envelope and can
	 * prefer `editdone` over another risky retry when it is about to run out.
	 */
	const buildImplementInjectMessage = (
		filepath: string,
		planText: string,
		idx: number,
		totalPlans: number,
		planStartedAt: number,
		budgetMs: number,
	): AgentMessage => {
		let content = "";
		try {
			const { abs, ok } = safeResolvePathUnderCwd(filepath);
			if (ok && existsSync(abs)) content = readFileSync(abs, "utf8");
		} catch {
			content = "";
		}
		const hasContent = content.length > 0;
		const numbered = hasContent ? formatFileWithZeroIndexedLines(content) : "(file does not yet exist — use `write` to create it)";
		// Paths recorded during PLAN mode (minus the current target file) — these
		// are the files the model already explored while planning and may want to
		// `read` again for surrounding-context before applying edits.
		const targetNorm = normalizePathForMatch(filepath);
		const contextPaths = planModeReadPaths.filter((p) => p !== targetNorm);
		const contextBlock = contextPaths.length > 0
			? `Paths already read during plan mode (**\`read\` only** — do not call \`edit\` or \`write\` on these):\n${contextPaths.map((p) => `- \`${p}\``).join("\n")}\n\n`
			: "";

		// Time-budget block. All numbers are whole seconds so the model can parse
		// them easily. We compute elapsed *now* so the block is always fresh.
		const now = Date.now();
		const planElapsedMs = Math.max(0, now - planStartedAt);
		const planRemainingMs = Math.max(0, budgetMs - planElapsedMs);
		const totalElapsedMs = Math.max(0, now - loopStart);
		const totalRemainingMs = Math.max(0, ULTIMATE_TIME_LIMIT_MS - totalElapsedMs);
		const toSec = (ms: number): string => (ms / 1000).toFixed(1);
		const warnLine =
			planRemainingMs <= PLAN_BUDGET_WARN_MS
				? `**WARNING**: only ${toSec(planRemainingMs)}s left in this plan's budget. `
				: "";
		const budgetBlock =
			`Time budget for THIS plan (plan ${idx + 1}/${totalPlans}):\n` +
			`- budget: ${toSec(budgetMs)}s\n` +
			`- elapsed on this plan: ${toSec(planElapsedMs)}s\n` +
			`- remaining in this plan: ${toSec(planRemainingMs)}s\n` +
			`- total run budget remaining (ultimate ${toSec(ULTIMATE_TIME_LIMIT_MS)}s cap): ${toSec(totalRemainingMs)}s\n` +
			(warnLine ? `${warnLine}\n` : "") +
			"\n";

		const body =
			`IMPLEMENT plan ${idx + 1}/${totalPlans}: \`${filepath}\`\n\n` +
			budgetBlock +
			`Plan for this file:\n---\n${planText || "(no plan text)"}\n---\n\n` +
			contextBlock +
			`Current full content of \`${filepath}\` (0-indexed line numbers — use them directly as startLine/endLine in \`edit\`):\n` +
			"```\n\n" +
			escapeMarkdownFences(numbered) +
			"\n```\n\n" +
			`Call exactly ONE of these tools:\n` +
			`- \`read\` — optional context from the paths above or elsewhere (**read-only** on non-plan files)\n` +
			`- \`edit\` — line-range replacements **only** on \`${filepath}\`\n` +
			`- \`write\` — full overwrite/create **only** for \`${filepath}\`\n` +
			`- \`editdone\` — signal this plan is complete (payload: { filepath: "${filepath}", plan: <plan text>, completedevidence: <short justification> })`;
		return {
			role: "user",
			content: [{ type: "text", text: body }],
			timestamp: ((Date.now() - loopStart) / 1000),
		};
	};

	/**
	 * Convenience wrapper: build the inject message for the **current** plan
	 * using the currently-tracked start time + a freshly-computed budget. All
	 * call-sites in implement mode funnel through this so the budget info is
	 * always consistent and up-to-date.
	 */
	const buildCurrentPlanInjectMessage = (): AgentMessage | null => {
		if (currentPlanIndex >= plannedOrder.length) return null;
		const p = plannedOrder[currentPlanIndex];
		if (currentPlanStartedAt === null) currentPlanStartedAt = Date.now();
		const budgetMs = computePlanBudgetMs();
		return buildImplementInjectMessage(
			p.path,
			p.plan,
			currentPlanIndex,
			plannedOrder.length,
			currentPlanStartedAt,
			budgetMs,
		);
	};

	const readCurrentFileContent = (filepath: string): string => {
		try {
			const { abs, ok } = safeResolvePathUnderCwd(filepath);
			if (ok && existsSync(abs)) return readFileSync(abs, "utf8");
		} catch {
			// best effort
		}
		return "";
	};

	const ensurePlanOriginalSnapshot = (planIdx: number): void => {
		if (planIdx < 0 || planIdx >= plannedOrder.length) return;
		if (originalContentByPlanIndex.has(planIdx)) return;
		const p = plannedOrder[planIdx];
		originalContentByPlanIndex.set(planIdx, readCurrentFileContent(p.path));
	};

	const buildEditdoneConfirmationInjectMessage = (
		planIdx: number,
		completedevidence: string,
	): AgentMessage | null => {
		if (planIdx < 0 || planIdx >= plannedOrder.length) return null;
		const p = plannedOrder[planIdx];
		const original = originalContentByPlanIndex.get(planIdx) ?? "";
		const edited = readCurrentFileContent(p.path);
		const originalNumbered = original.length > 0
			? formatFileWithZeroIndexedLines(original)
			: "(file did not exist when this plan started)";
		const editedNumbered = edited.length > 0
			? formatFileWithZeroIndexedLines(edited)
			: "(file currently missing)";
		const body =
			`IMPLEMENT plan ${planIdx + 1}/${plannedOrder.length} EDITDONE HANDSHAKE CONFIRMATION: \`${p.path}\`\n\n` +
			`You just called \`editdone\`. Confirm completion quality against this evidence before plan can advance.\n\n` +
			`Current plan text:\n---\n${p.plan || "(no plan text)"}\n---\n\n` +
			`First editdone completedevidence:\n---\n${completedevidence || "(empty)"}\n---\n\n` +
			`Original target file content at plan start:\n` +
			"```\n\n" +
			escapeMarkdownFences(originalNumbered) +
			"\n```\n\n" +
			`Edited target file content now:\n` +
			"```\n\n" +
			escapeMarkdownFences(editedNumbered) +
			"\n```\n\n" +
			`## IMPLEMENT MODE - EDITDONE QUALITY HANDSHAKE (NOT advanced yet)\n\n` +
			`Your first \`editdone\` call is treated as a draft completion claim only. The plan does NOT advance until you pass this self-audit and then call a second consecutive \`editdone\` with detailed evidence.\n\n` +
			`### Non-negotiable completion contract for this file\n` +
			`- **Plan lock:** You must satisfy the current plan exactly as written for \`${p.path}\`. Do not broaden scope; do not skip any promised behavior.\n` +
			`- **No hidden TODOs:** Any promised behavior that is not concretely implemented in the edited file is a failure of completion.\n` +
			`- **Correctness over optimism:** If any uncertainty remains, continue implementing now via \`read\`/\`edit\`/\`write\`; do not advance with a weak \`editdone\`.\n` +
			`- **Style consistency is mandatory:** Match local naming, formatting, control-flow style, error-handling conventions, and surrounding code patterns from the original file.\n` +
			`- **Critical:** Add blank line as my as possible. It doesn't reduce matched lines. When you add blank line, just COPY the blank line of original file.\n\n` +
			`### Mandatory self-review checklist (answer internally YES/NO)\n` +
			`1. **Plan coverage:** Did you implement every required edit in the plan text (not just part of it)?\n` +
			`2. **Behavioral completeness:** Are success path, failure path, and edge handling required by the plan actually present now?\n` +
			`3. **Code-level precision:** Are exact symbols, literals, branches, and data shapes aligned with what the plan demands?\n` +
			`4. **No regressions:** Did this change avoid breaking unrelated behavior in this file's existing logic?\n` +
			`5. **Style matching:** Does the edited file read like a natural continuation of the original coding style?\n` +
			`6. **Evidence quality:** Can you provide specific before/after evidence tied to concrete lines and logic changes (not vague claims)?\n` +
			`7. **Implement-mode discipline:** If any answer is NO, will you continue with a read/edit/write tool call now instead of forcing completion?\n\n` +
			`### Evidence standard for second \`editdone\`\n` +
			`Your second \`editdone\` \`completedevidence\` must be detailed and concrete. Include:\n` +
			`- exact changes made (symbols/blocks/branches touched)\n` +
			`- why those changes satisfy each required part of this plan\n` +
			`- why style and structure remain consistent with the original file\n` +
			`- why no remaining work is needed for this plan\n\n` +
			`### Call exactly ONE tool now\n` +
			`- If and only if all checklist items pass: call \`editdone\` again with detailed completedevidence. Only this second consecutive detailed \`editdone\` advances to the next plan.\n` +
			`- Otherwise: call \`read\`, \`edit\`, or \`write\` now to finish missing work for the current plan.\n`;
		return {
			role: "user",
			content: [{ type: "text", text: body }],
			timestamp: ((Date.now() - loopStart) / 1000),
		};
	};

	const isDetailedCompletionEvidence = (text: string): boolean => {
		const t = text.trim();
		if (t.length < 60) return false;
		const wordCount = t.split(/\s+/).filter((w) => w.length > 0).length;
		return wordCount >= 12;
	};

	/**
	 * Index into `currentContext.messages` that marks the "stable prefix" for
	 * implement mode — everything up to (but not including) the first implement-
	 * mode injection. We never trim below this index: this is the prompt that
	 * PLAN mode produced (task, discovery reads, `plan` tool call + result).
	 * Set to -1 while we are still in PLAN mode.
	 */
	let implementModeBaseMsgIdx = -1;

	/**
	 * Recognise an implement-mode inject message ("IMPLEMENT plan N/M: `path`…").
	 * All inject messages go through `buildImplementInjectMessage`, which always
	 * starts the body with that literal prefix — so this test is reliable.
	 */
	const isImplementInjectMessage = (m: AgentMessage): boolean => {
		if (m.role !== "user" || !Array.isArray(m.content)) return false;
		const first = m.content[0] as { type?: string; text?: unknown } | undefined;
		return typeof first?.text === "string" && (first.text as string).startsWith("IMPLEMENT plan ");
	};

	/**
	 * Within-plan trim: keep at most ONE implement-mode inject live at a time.
	 * Every re-inject on the current plan would otherwise stack another full
	 * file-content dump into history (~6k tokens each), driving prompt size
	 * from ~40k to 200k+ tokens and making every LLM call progressively slower.
	 * We remove any existing inject (both in the pending queue and already
	 * pushed into `currentContext.messages`) before queuing the new one.
	 *
	 * We do NOT touch assistant tool calls or their paired tool results, so the
	 * LLM's tool_use/tool_result invariant is preserved; only the stale user-
	 * message file dumps are discarded.
	 */
	const queueImplementInjectMessage = (msg: AgentMessage): void => {
		pendingMessages = pendingMessages.filter((m) => !isImplementInjectMessage(m));
		if (implementModeBaseMsgIdx >= 0) {
			for (let i = currentContext.messages.length - 1; i >= implementModeBaseMsgIdx; i--) {
				if (isImplementInjectMessage(currentContext.messages[i])) {
					currentContext.messages.splice(i, 1);
				}
			}
		}
		pendingMessages.push(msg);
	};

	/**
	 * Cross-plan wipe: when we advance past a completed (or budget-timed-out)
	 * plan, drop EVERYTHING from that plan's iteration out of
	 * `currentContext.messages`. We truncate back to `implementModeBaseMsgIdx`
	 * — the prompt used for the next LLM call is then the stable PLAN-mode
	 * prefix + the next plan's inject only.
	 *
	 * `newMessages` is intentionally left intact so the rollout / agent_end
	 * log still reflects everything the agent actually did.
	 */
	const resetToImplementModeBase = (reason: string, fromPlanIndex: number): void => {
		if (implementModeBaseMsgIdx < 0) return;
		const before = currentContext.messages.length;
		if (currentContext.messages.length > implementModeBaseMsgIdx) {
			currentContext.messages.length = implementModeBaseMsgIdx;
		}
		const pendingBefore = pendingMessages.length;
		pendingMessages = [];
		if (before > implementModeBaseMsgIdx || pendingBefore > 0) {
			void emitRolloutMarker("implement_context_trimmed", {
				reason,
				from_plan_index: fromPlanIndex,
				to_plan_index: currentPlanIndex,
				messages_dropped: before - implementModeBaseMsgIdx,
				pending_dropped: pendingBefore,
				base_idx: implementModeBaseMsgIdx,
			});
		}
	};

	type CriterionLedgerItem = {
		id: number;
		text: string;
		requiredFiles: string[];
		evidenceFiles: Set<string>;
	};

	const acceptanceCriteria: CriterionLedgerItem[] = [];

	// Parse expected files from system prompt discovery sections
	const parseExpectedFiles = (text: string): string[] => {
		const files: string[] = [];
		const seen = new Set<string>();
		const sectionPatterns = [
			/FILES EXPLICITLY NAMED IN THE TASK[^\n]*\n((?:[-*]\s+\S[^\n]*\n)+)/,
			/LIKELY RELEVANT FILES[^\n]*\n((?:[-*]\s+\S[^\n]*\n)+)/,
			/Pre-identified target files[^\n]*\n((?:[-*]\s+\S[^\n]*\n)+)/,
		];
		for (const re of sectionPatterns) {
			const match = text.match(re);
			if (!match) continue;
			const lineRe = /^[-*]\s+(\S[^(]*?)(?:\s+\(|\s*$)/gm;
			let m: RegExpExecArray | null;
			while ((m = lineRe.exec(match[1])) !== null) {
				const file = m[1].trim();
				if (file && !seen.has(file)) { seen.add(file); files.push(file); }
			}
		}
		return files;
	};
	const parseSectionFiles = (text: string, headerPattern: RegExp): string[] => {
		const files: string[] = [];
		const seen = new Set<string>();
		const match = text.match(headerPattern);
		if (!match) return files;
		const lineRe = /^[-*]\s+(\S[^(]*?)(?:\s+\(|\s*$)/gm;
		let m: RegExpExecArray | null;
		while ((m = lineRe.exec(match[1])) !== null) {
			const file = m[1].trim();
			if (file && !seen.has(file)) {
				seen.add(file);
				files.push(normalizePathForMatch(file));
			}
		}
		return files;
	};
	const parseExpectedFilesFromAcceptanceCriteria = (text: string): string[] => {
		const files: string[] = [];
		const seen = new Set<string>();
		// Prefer explicit inline-code paths in acceptance criteria bullets, e.g. `README.md`, `src/app.ts`.
		const codePathRe = /`([^`\n]+\.[A-Za-z0-9._/-]+)`/g;
		let m: RegExpExecArray | null;
		while ((m = codePathRe.exec(text)) !== null) {
			const candidate = m[1].trim();
			if (
				candidate.length > 0 &&
				!/^\d+$/.test(candidate) &&
				(candidate.includes("/") || candidate.includes("."))
			) {
				const norm = candidate.replace(/^\.?\//, "");
				if (!seen.has(norm)) {
					seen.add(norm);
					files.push(norm);
				}
			}
		}
		return files;
	};
	const parseExpectedCriteriaCount = (text: string): number => {
		const match = text.match(/This task has\s+(\d+)\s+acceptance criteria\./i);
		if (match) {
			const parsed = Number.parseInt(match[1], 10);
			if (Number.isFinite(parsed) && parsed > 0) return parsed;
		}
		// Fallback: count bullets only inside the Acceptance criteria block.
		const section = text.match(
			/Acceptance criteria:\s*\n([\s\S]*?)(?:\n\s*\n|\n[A-Z][^\n]*:|\n##|$)/i,
		);
		if (!section) return 0;
		const bullets = section[1].match(/^\s*(?:-\s+|\d+[.)]\s+)/gm);
		return bullets ? bullets.length : 0;
	};
	const parseAcceptanceCriteriaBulletCount = (text: string): number => {
		const section = text.match(
			/Acceptance criteria:\s*\n([\s\S]*?)(?:\n\s*\n|\n[A-Z][^\n]*:|\n##|$)/i,
		);
		if (!section) return 0;
		const bullets = section[1].match(/^\s*(?:-\s+|\d+[.)]\s+)/gm);
		return bullets ? bullets.length : 0;
	};
	const parseAcceptanceCriteriaBullets = (text: string): string[] => {
		const section = text.match(
			/Acceptance criteria:\s*\n([\s\S]*?)(?:\n\s*\n|\n[A-Z][^\n]*:|\n##|$)/i,
		);
		if (!section) return [];
		const lines = section[1].split("\n");
		const bullets: string[] = [];
		for (const line of lines) {
			const m = line.match(/^\s*(?:-\s+|\d+[.)]\s+)(.*)$/);
			if (!m) continue;
			const t = m[1].trim();
			if (t.length > 0) bullets.push(t);
		}
		return bullets;
	};
	const normalizePathForMatch = (p: string): string => p.replace(/^\.\//, "");
	const extractPlanItems = (raw: unknown): Array<{ path?: string; plan?: string; acceptance_criteria?: string[] }> => {
		if (Array.isArray(raw)) return raw as Array<{ path?: string; plan?: string }>;
		if (!raw || typeof raw !== "object") return [];
		const obj = raw as { plans?: unknown };
		return Array.isArray(obj.plans) ? (obj.plans as Array<{ path?: string; plan?: string; acceptance_criteria?: string[] }>) : [];
	};

	const buildPlanDraftEchoAgentResult = (rawArgs: unknown): AgentToolResult<{ planDraftEcho: true }> => {
		const collectOfficialTaskAcceptanceCriteriaLines = (): string[] => {
			const out: string[] = [];
			const seen = new Set<string>();
			const pushUnique = (lines: Iterable<string>) => {
				for (const raw of lines) {
					const t = raw.trim();
					if (t.length === 0 || seen.has(t)) continue;
					seen.add(t);
					out.push(t);
				}
			};
			pushUnique(acceptanceCriteria.map((c) => c.text));
			if (out.length === 0) {
				pushUnique(parseAcceptanceCriteriaBullets(systemPromptText));
			}
			const planPhasePrompt = currentContext.tauSystemPrompts?.plan;
			if (out.length === 0 && typeof planPhasePrompt === "string" && planPhasePrompt.length > 0) {
				pushUnique(parseAcceptanceCriteriaBullets(planPhasePrompt));
			}
			return out;
		};
		const officialCriteriaLines = collectOfficialTaskAcceptanceCriteriaLines();
		const plansArr = extractPlanItems(rawArgs);
		let appendix = "";
		try {
			const rawJson = JSON.stringify(rawArgs, null, 2);
			appendix =
				rawJson.length > PLAN_DRAFT_ECHO_JSON_MAX_CHARS
					? `${rawJson.slice(0, PLAN_DRAFT_ECHO_JSON_MAX_CHARS)}\n…(truncated)…`
					: rawJson;
		} catch {
			appendix = "(could not stringify plan payload)";
		}
		const planBlocks = plansArr.map((p, idx) => {
			const pathText = typeof p?.path === "string" ? p.path.trim() : "(missing path)";
			const ac = Array.isArray(p?.acceptance_criteria)
				? p.acceptance_criteria.filter((c): c is string => typeof c === "string" && c.trim().length > 0)
				: [];
			const body = typeof p?.plan === "string" ? p.plan.trim() : "";
			return (
				`#### Plan item ${idx + 1}: \`${pathText}\`\n` +
				(ac.length > 0 ? `Linked acceptance_criteria:\n${ac.map((c) => `- ${c}`).join("\n")}\n\n` : "") +
				`${body || "(empty plan text)"}\n`
			);
		});
		const officialCriteriaBlock =
			officialCriteriaLines.length > 0
				? officialCriteriaLines.map((c, i) => `${i + 1}. ${c}`).join("\n")
				: "(The agent could not extract a bullet list under an **Acceptance criteria:** section from the task / system prompt. Treat the task message you were given as the source of truth: copy its criteria **verbatim** into your next `plan` payload and map every one to a file plan.)";
		const text =
			`## PLAN MODE — DRAFT HANDSHAKE (NOT validated)\n\n` +
			`Your \`plan\` tool call was received. **The agent did not run path checks, file existence checks, or acceptance-criteria validation on this submission.**\n\n` +
			`This is intentional: your first job is to **brutally self-audit** your draft against the **official** task acceptance criteria below (from the injected task / system prompt — **not** taken from your tool arguments). Do not trust the strings you put in \`task_acceptance_criteria\` until they match this list **exactly**.\n\n` +
			`### IMPLEMENT-mode contract (non-negotiable — read as law)\n` +
			`After your **validated** second \`plan\`-only submission, the agent enters **IMPLEMENT** mode. There you must **complete every official acceptance criterion** using **only** what is already frozen in your plans — **not** by treating implement mode as a second discovery pass.\n\n` +
			`- **No “research to finish the spec”:** You must **not** depend on exploratory \`grep\`, broad \`read\` sweeps, or guessing owners/symbols to figure out what the task meant. Anything required to pass a criterion must already appear in the matching \`plans[].plan\` (**Scope / Edits / Acceptance / Verification**).\n` +
			`- **Reads are local glue only:** In implement mode, \`read\` exists only to refresh exact file text or skim **already-scoped** neighbors (imports, small helpers) for the **current** planned file — **not** to replace missing \`Edits:\` detail you should have written here in PLAN mode.\n` +
			`- **Path proof policy (NON-NEGOTIABLE):** every path used in \`read\` and every \`plans[].path\` must be an **EXACT verbatim discovery-proven path** copied from \`ls\`/\`find\`/\`grep\`/\`bash\` output. Guessed or inferred paths are invalid.\n` +
			`- **Path-not-found handling:** if \`read\` fails with path-not-found, re-run discovery first and copy the proven path character-for-character; never retry with another guessed variant.\n` +
			`- **Every \`Edits:\` bullet is an executable contract:** Each bullet must be **self-contained**: a competent engineer (or another LLM) can perform that edit **without opening unrelated files to infer intent**. Name **concrete** types, methods, fields, routes, HTTP status codes, JSON property names, zip glob patterns, directories, and before→after behavior.\n` +
			`- **Ban vague edit language:** Phrases like “wire up appropriately”, “ensure authentication works”, “handle errors”, “update as needed”, “integrate with the system”, or “add support for X” **without** naming the exact symbol and change are **forbidden** — rewrite into precise mechanical steps.\n\n` +
			`### OFFICIAL task acceptance criteria (from the task)\n${officialCriteriaBlock}\n\n` +
			`### Mandatory self-review (answer internally with YES/NO — any NO means you are NOT ready to commit)\n` +
			`- **Completeness vs IMPLEMENT:** Could **every** official criterion above be satisfied **solely** by executing your \`Edits:\` bullets as written, with at most **local** reads on the planned path — with **zero** extra discovery?\n` +
			`- **Completeness of mapping:** Does your \`plans[]\` set **fully** cover **every** official criterion, with **no** criterion implicit, partial, or “mostly” addressed?\n` +
			`- **Zero ambiguity:** Does **each** \`Edits:\` line name **exact** symbols (types / methods / fields / routes) and the **exact** behavioral change — no room for interpretation?\n` +
			`- **Bullet clarity:** Is every edit bullet **one clear action** (or a tight numbered sub-list) that a stranger could implement in one pass through \`plans[].path\`?\n` +
			`- **Implementation density:** Are edge cases, error paths, and verification steps for **this file** spelled out in \`Edits:\` / \`Verification:\` — not deferred to “figure out during implement”?\n` +
			`- **Verbatim JSON:** Will your next \`plan\` payload’s \`task_acceptance_criteria\` array be a **verbatim** copy of the official lines below (same strings, same count — no paraphrase, no invented extras)?\n` +
			`### Your planned files (human-readable — your draft payload)\n\n` +
			(planBlocks.length > 0 ? planBlocks.join("\n---\n\n") : "(no plans[] entries parsed from payload)\n") +
			`\n### Full submitted JSON (for diffing; may be truncated)\n\`\`\`json\n${escapeMarkdownFences(appendix)}\n\`\`\`\n\n` +
			`---\n` +
			`### SELF-AUDIT CHECKLIST (answer every point internally before your next action)\n` +
			`1. **Criterion → bullet trace (NON-NEGOTIABLE):** For **each** official criterion, list the **exact** \`Edits:\` bullet(s) (quote their opening phrase) that implement it, and the \`plans[].path\` for each. If any criterion lacks a bullet-level trace, your plan is **incomplete**.\n` +
			`2. **Implement-without-discovery test:** For each plan item, cover the \`path\` column on your screen and ask: “Could I implement **only** from \`Scope\`+\`Edits\`+\`Acceptance\`+\`Verification\`?” If **no**, expand \`Edits:\` until **yes**.\n` +
			`3. **Right file, right symbol:** Every symbol named in \`Edits:\` for an item must **live in** that item’s \`path\` (or be a type explicitly imported there). No “edit method M in file F” when M is actually in another file.\n` +
			`4. **Literal fidelity:** Every task literal that affects runtime (paths, globs like \`devtools*.zip\`, HTTP codes, directory names like **debug**, strings) appears **verbatim** in the relevant plan text where it matters.\n` +
			`5. **Verbatim proven paths:** Every \`plans[].path\` is copied character-for-character from proven discovery output (\`ls\`/\`find\`/\`grep\`/\`bash\`). No guessed folders, no guessed filenames, no “probably this path”.\n` +
			`6. **Ordering:** \`plans[]\` is **dependency / leaf-first** (new leaves before consumers, wiring last) for sequential implement mode.\n` +
			`7. **New vs existing files:** \`is_new_file\` matches disk reality for every path.\n\n` +
			`---\n` +
			`### WHAT TO DO NEXT (strict handshake)\n` +
			`- If you still need **repository evidence to write those concrete bullets**, call **\`read\` / \`grep\` / \`find\` / \`ls\` / \`bash\`** **now** in PLAN mode. **Any turn that includes a tool other than \`plan\` resets this handshake** — your next \`plan\`-only turn becomes a fresh draft echo.\n` +
			`- If you are satisfied after self-audit, call **only** \`plan\` again on a **later turn** with the same or corrected JSON. **That second consecutive \`plan\`-only turn is validated** and freezes the contract for IMPLEMENT mode.\n` +
			`- Do not rely on prose outside tools; the draft echo is not a substitute for a validated plan.\n`;

		return {
			content: [{ type: "text", text }],
			details: { planDraftEcho: true as const },
		};
	};

	const executePlanModeToolBatchWithHandshake = async (
		assistantMessage: AssistantMessage,
		calls: AgentToolCall[],
	): Promise<ToolResultMessage[]> => {
		const injectPlanModeReadPaths = (tc: AgentToolCall): AgentToolCall => {
			if (tc.name !== "plan") return tc;
			const args = (tc.arguments && typeof tc.arguments === "object")
				? { ...(tc.arguments as Record<string, unknown>) }
				: {};
			args.plan_mode_read_paths = [...planModeReadPaths];
			return { ...tc, arguments: args };
		};
		const hasNonPlan = calls.some((tc) => tc.name !== "plan");
		if (planHandshakeAwaitingConsecutivePlanOnly && hasNonPlan) {
			planHandshakeAwaitingConsecutivePlanOnly = false;
			await emitRolloutMarker("plan_handshake_reset", {
				reason: "non_plan_tool_in_turn",
				tools: calls.map((t) => t.name),
			});
		}

		const hasPlan = calls.some((tc) => tc.name === "plan");
		const onlyPlan = hasPlan && calls.every((tc) => tc.name === "plan");
		const planBypassExceeded = (Date.now() - loopStart) >= PLAN_MODE_BYPASS_MS;
		// Timeout override: once PLAN budget is exceeded, a plan-only turn should
		// execute real plan validation immediately (skip draft-confirmation gate).
		const useRealPlanValidation =
			onlyPlan && (planHandshakeAwaitingConsecutivePlanOnly || planBypassExceeded);

		const results: ToolResultMessage[] = [];
		for (const toolCall of calls) {
			const toolCallWithReadPaths = injectPlanModeReadPaths(toolCall);
			await emit({
				type: "tool_execution_start",
				toolCallId: toolCallWithReadPaths.id,
				toolName: toolCallWithReadPaths.name,
				args: toolCallWithReadPaths.arguments,
			});

			if (toolCallWithReadPaths.name === "plan" && !useRealPlanValidation) {
				const tc = applyRobustWorkspacePathsToToolCall(toolCallWithReadPaths);
				const rawPlanArgs = tc.arguments ?? {};
				await emitRolloutMarker("plan_tool_args", {
					args: rawPlanArgs,
				});
				const plansEcho = extractPlanItems(rawPlanArgs);
				await emitRolloutMarker("plan_draft_echo", {
					plan_count: plansEcho.length,
				});
				const synthetic = buildPlanDraftEchoAgentResult(rawPlanArgs);
				results.push(await emitToolCallOutcome(tc, synthetic, false, emit, loopStart));
				continue;
			}

			const tcExec = applyRobustWorkspacePathsToToolCall(toolCallWithReadPaths);
			if (tcExec.name === "plan" && useRealPlanValidation) {
				await emitRolloutMarker("plan_tool_args", {
					args: tcExec.arguments ?? {},
				});
			}

			const preparation = await prepareToolCall(currentContext, assistantMessage, tcExec, config, signal);
			if (preparation.kind === "immediate") {
				results.push(await emitToolCallOutcome(tcExec, preparation.result, preparation.isError, emit, loopStart));
			} else {
				const executed = await executePreparedToolCall(preparation, signal, emit);
				results.push(
					await finalizeExecutedToolCall(
						currentContext,
						assistantMessage,
						preparation,
						executed,
						config,
						signal,
						emit,
						loopStart,
					),
				);
			}
		}

		if (onlyPlan && !useRealPlanValidation) {
			planHandshakeAwaitingConsecutivePlanOnly = true;
			await emitRolloutMarker("plan_handshake_awaiting_validated_resubmit", {});
		}

		return results;
	};
	const isMutatingBashCommand = (command: string): boolean => {
		const cmd = command.trim();
		if (!cmd) return false;
		// Conservative mutation detection for PLAN mode: block known write/destructive shell patterns.
		return (
			/(^|[;&|]\s*)(rm|mv|cp|touch|mkdir|rmdir|truncate|install|chmod|chown)\b/.test(cmd) ||
			/(^|[;&|]\s*)(git\s+(apply|am|commit|checkout|switch|reset|clean)\b)/.test(cmd) ||
			/(^|[;&|]\s*)(sed\s+-i\b|perl\s+-i\b|python\s+-c\b|node\s+-e\b)/.test(cmd) ||
			/(^|[;&|]\s*)(tee\b|dd\b)/.test(cmd) ||
			/(^|[;&|]\s*)(npm\s+(install|i|ci)\b|pnpm\s+(install|i|add)\b|yarn\s+(install|add)\b)/.test(cmd) ||
			/(>>?|<<?)\s*\S+/.test(cmd)
		);
	};
	const isNetworkProbeBashCommand = (command: string): boolean => {
		const cmd = command.trim();
		if (!cmd) return false;
		return (
			/\b(curl|wget|nc|netcat|telnet|ping)\b/.test(cmd) ||
			/\bpython\d?\s+-c\b/.test(cmd) ||
			/\bnode\s+-e\b/.test(cmd) ||
			/\b(socket|127\.0\.0\.1|localhost|http:\/\/|https:\/\/)\b/.test(cmd)
		);
	};
	const emitRolloutMarker = async (name: string, payload: Record<string, unknown> = {}): Promise<void> => {
		const marker = {
			type: name,
			ts_ms: Date.now(),
			...payload,
		};
		pendingTurnEndMarkerLines.push(JSON.stringify(marker));
		const markerMessage: AgentMessage = {
			role: "user",
			content: [{ type: "text", text: `[agent_loop_event] ${JSON.stringify(marker)}` }],
			timestamp: ((Date.now() - loopStart) / 1000),
		};
		await emit({ type: "message_start", message: markerMessage });
		await emit({ type: "message_end", message: markerMessage });
	};
	const criterionNeedsFile = (criterion: CriterionLedgerItem, filePath: string): boolean => {
		const f = normalizePathForMatch(filePath);
		return criterion.requiredFiles.some((rf) => {
			const r = normalizePathForMatch(rf);
			return f === r || f.endsWith("/" + r) || r.endsWith("/" + f);
		});
	};
	const updateCriterionLedgerWithEdit = (targetPath: string): void => {
		const norm = normalizePathForMatch(targetPath);
		for (const c of acceptanceCriteria) {
			if (c.requiredFiles.length === 0) continue;
			if (criterionNeedsFile(c, norm)) c.evidenceFiles.add(norm);
		}
	};
	const missingPlannedFiles = (): string[] => {
		const missing: string[] = [];
		for (const pf of plannedFiles) {
			const p = normalizePathForMatch(pf);
			let touched = false;
			for (const e of editedPaths) {
				const en = normalizePathForMatch(e);
				if (en === p || en.endsWith("/" + p) || p.endsWith("/" + en)) {
					touched = true;
					break;
				}
			}
			if (!touched) missing.push(pf);
		}
		return missing;
	};
	// PLAN timing policy (exactly two thresholds):
	// 1) 100s warning/escalation
	// 2) 150s bypass (accept latest plan turn without strict confirmation gate)
	const PLAN_MODE_WARNING_MS = 100_000;
	const PLAN_MODE_BYPASS_MS = 150_000;
	const IMPLEMENT_VERIFY_MAX_MS = 200_000;
	const IMPLEMENT_READ_ONLY_REQUIRED_TURN_LIMIT = 3;

	const hasFullReadForPath = (path: string): boolean => {
		const norm = normalizePathForMatch(path);
		return pathsAlreadyRead.has(norm) || pathsAlreadyRead.has("./" + norm);
	};
	const hasReadEvidenceForExpectedPath = (expectedPath: string): boolean => {
		const norm = normalizePathForMatch(expectedPath);
		if (hasFullReadForPath(norm)) return true;
		// For basename-only expectations (e.g. "README.md"), accept any read path that ends with that basename.
		// This avoids deadlocks when discovery extracts shorthand names while reads use full relative paths.
		if (!norm.includes("/")) {
			for (const rp of pathsAlreadyRead) {
				const rpn = normalizePathForMatch(rp);
				const base = rpn.split("/").pop() ?? rpn;
				if (base === norm) return true;
			}
		}
		return false;
	};

	// Extract expected files from system prompt or initial messages
	const systemPromptText = (currentContext as any).systemPrompt || "";
	let expectedFiles: string[] = parseExpectedFiles(systemPromptText);
	let explicitNamedFiles: string[] = parseSectionFiles(
		systemPromptText,
		/FILES EXPLICITLY NAMED IN THE TASK[^\n]*\n((?:[-*]\s+\S[^\n]*\n)+)/,
	);
	for (const f of parseExpectedFilesFromAcceptanceCriteria(systemPromptText)) {
		if (!expectedFiles.includes(f)) expectedFiles.push(f);
	}
	let expectedCriteriaCount = parseExpectedCriteriaCount(systemPromptText);
	if (expectedFiles.length === 0) {
		for (const msg of currentContext.messages) {
			if (!("content" in msg) || !Array.isArray(msg.content)) continue;
			for (const block of msg.content as any[]) {
				if (block?.type === "text" && typeof block.text === "string") {
					const parsed = parseExpectedFiles(block.text);
					if (parsed.length > 0) { expectedFiles = parsed; break; }
				}
			}
			if (expectedFiles.length > 0) break;
		}
	}
	if (explicitNamedFiles.length === 0) {
		for (const msg of currentContext.messages) {
			if (!("content" in msg) || !Array.isArray(msg.content)) continue;
			for (const block of msg.content as any[]) {
				if (block?.type === "text" && typeof block.text === "string") {
					const parsed = parseSectionFiles(
						block.text,
						/FILES EXPLICITLY NAMED IN THE TASK[^\n]*\n((?:[-*]\s+\S[^\n]*\n)+)/,
					);
					if (parsed.length > 0) {
						explicitNamedFiles = parsed;
						break;
					}
				}
			}
			if (explicitNamedFiles.length > 0) break;
		}
	}
	// Even when discovery sections are absent, acceptance criteria often name explicit files in backticks.
	for (const msg of currentContext.messages) {
		if (!("content" in msg) || !Array.isArray(msg.content)) continue;
		for (const block of msg.content as any[]) {
			if (block?.type === "text" && typeof block.text === "string") {
				for (const f of parseExpectedFilesFromAcceptanceCriteria(block.text)) {
					if (!expectedFiles.includes(f)) expectedFiles.push(f);
				}
			}
		}
	}
	if (expectedCriteriaCount === 0) {
		for (const msg of currentContext.messages) {
			if (!("content" in msg) || !Array.isArray(msg.content)) continue;
			for (const block of msg.content as any[]) {
				if (block?.type === "text" && typeof block.text === "string") {
					const parsed = parseExpectedCriteriaCount(block.text);
					if (parsed > 0) {
						expectedCriteriaCount = parsed;
						break;
					}
				}
			}
			if (expectedCriteriaCount > 0) break;
		}
	}
	if (expectedCriteriaCount === 0) {
		expectedCriteriaCount = parseAcceptanceCriteriaBulletCount(systemPromptText);
	}
	{
		const allCriteriaTexts = new Set<string>();
		for (const t of parseAcceptanceCriteriaBullets(systemPromptText)) allCriteriaTexts.add(t);
		for (const msg of currentContext.messages) {
			if (!("content" in msg) || !Array.isArray(msg.content)) continue;
			for (const block of msg.content as any[]) {
				if (block?.type === "text" && typeof block.text === "string") {
					for (const t of parseAcceptanceCriteriaBullets(block.text)) allCriteriaTexts.add(t);
				}
			}
		}
		let cid = 1;
		for (const text of allCriteriaTexts) {
			const requiredFiles = parseExpectedFilesFromAcceptanceCriteria(text);
			acceptanceCriteria.push({ id: cid++, text, requiredFiles, evidenceFiles: new Set<string>() });
		}
	}
	if (expectedFiles.length > 0) {
		foundFiles = [...expectedFiles];
	}
	const requiredPlanPathsForTransition = (): string[] => {
		const required = new Set<string>();
		for (const f of explicitNamedFiles) {
			const n = normalizePathForMatch(f);
			if (n.includes("/")) required.add(n);
		}
		for (const c of acceptanceCriteria) {
			for (const f of c.requiredFiles) {
				const n = normalizePathForMatch(f);
				if (n.includes("/")) required.add(n);
			}
		}
		return [...required];
	};
	const missingRequiredFromPlan = (submittedPlanPaths: Set<string>): string[] => {
		const required = requiredPlanPathsForTransition();
		if (required.length === 0) return [];
		const missing: string[] = [];
		for (const r of required) {
			let covered = false;
			for (const p of submittedPlanPaths) {
				if (p === r || p.endsWith("/" + r) || r.endsWith("/" + p)) {
					covered = true;
					break;
				}
			}
			if (!covered) missing.push(r);
		}
		return missing;
	};
	const undiscoveredExpectedFiles = (): string[] => {
		if (expectedFiles.length === 0) return [];
		const missing: string[] = [];
		for (const f of expectedFiles) {
			const norm = normalizePathForMatch(f);
			// Require discovery only for concrete path-like expectations.
			if (!norm.includes("/")) continue;
			// Only gate on files that actually exist in workspace; avoid impossible reads.
			const { abs, ok } = safeResolvePathUnderCwd(norm);
			if (!ok || !existsSync(abs)) continue;
			if (!hasReadEvidenceForExpectedPath(norm)) {
				missing.push(f);
			}
		}
		return missing;
	};
	const GRACEFUL_EXIT_MS = 170_000;
	/** If identical tool-call signatures repeat with no new successful mutations, force-stop to preserve diff. */
	const REPEATED_TOOL_SIGNATURE_LIMIT = 3;
	let multiFileHintSent = false;
	let successfulMutationCount = 0;
	let lastToolSignature = "";
	let repeatedToolSignatureCount = 0;
	let lastSignatureMutationCount = 0;
	const baselineFileContentByPath = new Map<string, string>();
	const netChangedPaths = new Set<string>();
	const siblingHintedReadDirs = new Set<string>();
	let implementModeStartedAt: number | null = null;
	/** Successful `edit` or `write` mutates disk — both must advance scoring-related loop state (was edit-only). */
	const recordSuccessfulFileMutation = async (targetPath: string): Promise<void> => {
		editFailMap.set(targetPath, 0);
		editNotFoundStreakMap.set(targetPath, 0);
		priorFailedAnchor.delete(targetPath);
		successfulMutationCount++;
		const firstMutation = !hasProducedEdit;
		hasProducedEdit = true;
		explorationCount = 0;
		const normTarget = targetPath.replace(/^\.\//, "");
		updateCriterionLedgerWithEdit(normTarget);
		editedPaths.add(targetPath);
		editedPaths.add(normTarget);
		editedPaths.add("./" + normTarget);
		pathEditCounts.set(normTarget, (pathEditCounts.get(normTarget) ?? 0) + 1);
		if (normTarget === lastEditedFile) {
			consecutiveEditsOnSameFile++;
		} else {
			consecutiveEditsOnSameFile = 1;
			lastEditedFile = normTarget;
		}
		pendingMessages.push({
			role: "user",
			content: [
				{
					type: "text",
					text: `\`${targetPath}\` updated successfully.`,
				},
			],
			timestamp: ((Date.now() - loopStart) / 1000),
		});
		if (firstMutation && !multiFileHintSent && plannedFiles.size >= 2) {
			multiFileHintSent = true;
			const remainingPlanned = missingPlannedFiles();
			if (remainingPlanned.length > 0) {
				pendingMessages.push({
					role: "user",
					content: [
						{
							type: "text",
							text: `Plan-scoped reminder: remaining planned files not yet edited: ${remainingPlanned.slice(0, 6).map((f) => `\`${f}\``).join(", ")}. Continue implementing planned files only — do NOT edit off-plan files even if they seem related.`,
						},
					],
					timestamp: ((Date.now() - loopStart) / 1000),
				});
			}
		}
	};
	const refreshNetChangedStateForPath = (targetPath: string): void => {
		const norm = normalizePathForMatch(targetPath);
		if (!norm) return;
		const baseline = baselineFileContentByPath.get(norm);
		if (typeof baseline !== "string") return;
		try {
			const { abs, ok } = safeResolvePathUnderCwd(norm);
			if (!ok || !existsSync(abs)) return;
			const current = readFileSync(abs, "utf8");
			if (current === baseline) netChangedPaths.delete(norm);
			else netChangedPaths.add(norm);
		} catch {
			// best-effort only
		}
	};
	const maybeAddSiblingHintAfterRead = async (readPath: string): Promise<void> => {
		if (executionMode !== "plan") return;
		const normRead = readPath.replace(/^\.\//, "");
		const dir = normRead.includes("/") ? normRead.substring(0, normRead.lastIndexOf("/")) : ".";
		if (siblingHintedReadDirs.has(dir)) return;
		try {
			const { spawnSync: _sibSpawn } = await import("node:child_process");
			const lsResult = _sibSpawn("ls", [dir], { cwd: process.cwd(), timeout: 1000, encoding: "utf-8" });
			if (lsResult.status !== 0 || !lsResult.stdout) return;
			const siblings = lsResult.stdout
				.trim()
				.split("\n")
				.map((f: string) => (dir === "." ? f : dir + "/" + f))
				.filter((f: string) => f && f !== normRead);
			const codeExts = new Set([".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".rs", ".dart", ".vue", ".svelte", ".rb", ".java", ".kt", ".cs", ".cpp", ".c", ".h", ".php", ".swift"]);
			const related = siblings
				.filter((f: string) => {
					const name = f.split("/").pop() || "";
					const fext = name.includes(".") ? "." + name.split(".").pop() : "";
					return codeExts.has(fext) || name.includes(".test.") || name.includes(".spec.") || name.includes(".freezed.");
				})
				.slice(0, 8);
			if (related.length === 0) return;
			for (const rf of related) {
				if (!foundFiles.includes(rf)) foundFiles.push(rf);
			}
			siblingHintedReadDirs.add(dir);
			pendingMessages.push({
				role: "user",
				content: [{
					type: "text",
					text: `Siblings near \`${normRead}\`: ${related.map((f: string) => `\`${f}\``).join(", ")}. In PLAN mode, read only those that map to acceptance criteria before submitting \`plan\`.`,
				}],
				timestamp: ((Date.now() - loopStart) / 1000),
			});
		} catch {
			return;
		}
	};

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	// Optional git hint (from v701): merge paths that differ vs a base ref into expected targets.
	// Unlike v701, we do not delete paths — only broaden coverage for nudges.
	try {
		const { spawnSync: _gSpawn } = await import("node:child_process");
		const _cwd = process.cwd();
		const _git = (args: string[]) => {
			try {
				const r = _gSpawn("git", args, { cwd: _cwd, timeout: 3000, encoding: "utf-8" });
				return r.status === 0 ? (r.stdout || "").trim() : "";
			} catch {
				return "";
			}
		};
		const _head = _git(["rev-parse", "HEAD"]);
		const _refs = _git(["for-each-ref", "--format=%(objectname)%09%(refname)"]);
		if (_head && _refs) {
			let _refSha = "";
			for (const _line of _refs.split("\n")) {
				const [_sha, _name] = _line.split("\t");
				if (_sha && _sha !== _head && _name && (_name.includes("/main") || _name.includes("/master"))) {
					_refSha = _sha;
					break;
				}
			}
			if (!_refSha) {
				for (const _line of _refs.split("\n")) {
					const [_sha, _name] = _line.split("\t");
					if (_sha && _sha !== _head && _name) {
						_refSha = _sha;
						break;
					}
				}
			}
			if (_refSha) {
				const _dt = _git(["diff-tree", "--raw", "--no-renames", "-r", _head, _refSha]);
				const _rf: string[] = [];
				for (const _dl of _dt.split("\n")) {
					const _dm = _dl.match(/^:\d+ \d+ [0-9a-f]+ [0-9a-f]+ ([AMD])\t(.+)$/);
					if (!_dm) continue;
					if (_dm[1] === "A" || _dm[1] === "M") _rf.push(_dm[2]);
				}
				if (_rf.length > 0 && _rf.length <= 20) {
					const _norm = (s: string) => s.replace(/^\.\//, "");
					let toMerge = _rf;
					if (expectedFiles.length > 0) {
						toMerge = _rf.filter((p) => {
							const np = _norm(p);
							return expectedFiles.some((e) => {
								const ne = _norm(e);
								return np === ne || np.endsWith("/" + ne) || ne.endsWith("/" + np);
							});
						});
					}
					if (toMerge.length > 0) {
						const merged = new Set([...foundFiles, ...toMerge, ...expectedFiles]);
						foundFiles = [...merged];
						expectedFiles = [...merged];
					}
				}
			}
		}
	} catch {
		/* not a git repo or git unavailable */
	}
	await emitRolloutMarker("mode_transition", { from: null, to: "plan", reason: "run_start" });
	const isPlanSteeringMessage = (message: AgentMessage): boolean => {
		if (!("content" in message) || !Array.isArray((message as any).content)) return false;
		const text = (message as any).content
			.filter((c: any) => c?.type === "text")
			.map((c: any) => c?.text ?? "")
			.join("\n");
		return (
			text.includes("PLAN search protocol") ||
			text.includes("Follow right-search order now") ||
			text.includes("Plan mode timeout reached (~100s)")
		);
	};
	if (executionMode === "plan" && pendingMessages.length === 0) {
		pendingMessages.push({
			role: "user",
			content: [{
				type: "text",
				text: "PLAN search protocol (follow in order): 1) grep exact task terms/paths/symbols, 2) read top owner file(s), 3) sweep only required wiring neighbors (entrypoint/coordinator/utility/fallback/interface), 4) verify every acceptance criterion maps to file(s), 5) call `plan` immediately with exact JSON format. Avoid broad unfocused exploration.",
			}],
			timestamp: ((Date.now() - loopStart) / 1000),
		});
	}
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				await emit({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			// Ultimate wall-clock guard: once ULTIMATE_TIME_LIMIT_MS has elapsed since
			// the start of the run, end the agent *successfully* so whatever edits
			// have been landed so far are returned as a partial solution instead of
			// being lost to an external harness hard-kill (Docker 300s, etc.).
			if ((Date.now() - loopStart) >= ULTIMATE_TIME_LIMIT_MS) {
				await emitRolloutMarker("ultimate_time_limit_reached", {
					elapsed_ms: Date.now() - loopStart,
					limit_ms: ULTIMATE_TIME_LIMIT_MS,
					completed_plans: currentPlanIndex,
					total_plans: plannedOrder.length,
				});
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// Process pending messages (inject before next assistant response)
			if (pendingMessages.length > 0) {
				{
					pendingMessages = pendingMessages.filter((m) => !isPlanSteeringMessage(m));
				}
				for (const message of pendingMessages) {
					await emit({ type: "message_start", message });
					await emit({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			// Stream assistant response
			const llmSystemPrompt =
				currentContext.tauSystemPrompts != null
					? executionMode === "plan"
						? currentContext.tauSystemPrompts.plan
						: currentContext.tauSystemPrompts.implement
					: currentContext.systemPrompt;

			const planWarningExceeded =
				executionMode === "plan" && !planSubmitted && (Date.now() - loopStart) >= PLAN_MODE_WARNING_MS;

			const message = await streamAssistantResponse(
				currentContext,
				config,
				signal,
				emit,
				streamFn,
				{
					// PLAN mode: force a tool call (once past the time budget, force `plan` specifically).
					// IMPLEMENT mode: force a tool call so the model must emit edit / write / editdone.
					toolChoice:
						executionMode === "plan" && !planSubmitted
							? planWarningExceeded
								? ({ type: "function", function: { name: "plan" } } as const)
								: ("required" as const)
							: executionMode === "implement"
								? ("required" as const)
								: undefined,
				},
				llmSystemPrompt,
			);
			newMessages.push(message);

			if (message.stopReason === "aborted") {
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			if (message.stopReason === "error") {
				if (upstreamRetries < UPSTREAM_RETRY_LIMIT) {
					upstreamRetries++;
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push({
						role: "user",
						content: [
							{
								type: "text",
								text: "Transient upstream failure occurred. Resume by calling a tool directly — avoid prose. Only file diffs count toward your evaluation score.",
							},
						],
						timestamp: ((Date.now() - loopStart) / 1000),
					});
					hasMoreToolCalls = false;
					continue;
				}
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			const toolCalls = message.content.filter((c) => c.type === "toolCall");
			// Gemini sometimes hallucinates "EditEdits" or "editEdits" instead of "edit"
			for (const tc of toolCalls) {
				if (tc.name === "EditEdits" || tc.name === "editEdits") {
					(tc as { name: string }).name = "edit";
				}
			}
			hasMoreToolCalls = toolCalls.length > 0;
			const hasPlanToolCall = toolCalls.some((tc) => tc.name === "plan");
			const toolResults: ToolResultMessage[] = [];
			if (executionMode === "plan") {
				if (hasPlanToolCall && expectedFiles.length > 0 && !planWarningExceeded) {
					const undiscovered = undiscoveredExpectedFiles();
					if (undiscovered.length > 0) {
						await emit({ type: "turn_end", message, toolResults: [] });
						pendingMessages.push({
							role: "user",
							content: [{
								type: "text",
								text: `Before calling \`plan\`, discover expected files first. Call \`read\` (full file) on: ${undiscovered
									.slice(0, 12)
									.map((f) => `\`${f}\``)
									.join(", ")}. Path policy: use only EXACT verbatim paths proven by \`ls\`/\`find\`/\`grep\` output; do not guess paths.`,
							}],
							timestamp: ((Date.now() - loopStart) / 1000),
						});
						hasMoreToolCalls = false;
						continue;
					}
				}
				{
					const hasMutationAttempt = toolCalls.some((tc) => {
						if (tc.name === "edit" || tc.name === "write") return true;
						if (tc.name !== "bash") return false;
						const cmd = String((tc.arguments as { command?: string } | undefined)?.command ?? "");
						return isMutatingBashCommand(cmd);
					});
					const hasNetworkProbe = toolCalls.some((tc) => {
						if (tc.name !== "bash") return false;
						const cmd = String((tc.arguments as { command?: string } | undefined)?.command ?? "");
						return isNetworkProbeBashCommand(cmd);
					});
					if (hasMutationAttempt) {
						await emit({ type: "turn_end", message, toolResults: [] });
						pendingMessages.push({
							role: "user",
							content: [{ type: "text", text: "Still in PLAN mode. File mutations are forbidden before plan submission (`edit`/`write` and mutating `bash` commands). Continue broad discovery and then call `plan` with exact JSON format." }],
							timestamp: ((Date.now() - loopStart) / 1000),
						});
						hasMoreToolCalls = false;
						continue;
					}
					if (hasNetworkProbe) {
						await emit({ type: "turn_end", message, toolResults: [] });
						pendingMessages.push({
							role: "user",
							content: [{ type: "text", text: "PLAN mode forbids network/probe bash commands (they often fail with ConnectionRefusedError in eval sandboxes). Use only local repository discovery tools and then call `plan` with exact JSON format." }],
							timestamp: ((Date.now() - loopStart) / 1000),
						});
						hasMoreToolCalls = false;
						continue;
					}
				}
				if ((Date.now() - loopStart) >= PLAN_MODE_WARNING_MS && pendingMessages.length === 0 && !hasPlanToolCall) {
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push({
						role: "user",
						content: [{ type: "text", text: "Plan mode timeout reached (~100s). Stop exploration now. Do one final coverage check (each criterion -> file) and call only the `plan` tool with detailed JSON plans." }],
						timestamp: ((Date.now() - loopStart) / 1000),
					});
					hasMoreToolCalls = false;
					continue;
				}
				if (!hasMoreToolCalls) {
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push({
						role: "user",
						content: [
							{
								type: "text",
								text: "You are still in PLAN mode. Your last turn had no effective planning progress. Follow right-search order now: grep exact task terms -> read owner file(s) -> sweep required wiring neighbors -> criterion-to-file coverage check -> call `plan` with exact JSON format.",
							},
						],
						timestamp: ((Date.now() - loopStart) / 1000),
					});
					continue;
				}
				{
					const planAllowed = new Set(["bash", "find", "grep", "ls", "read", "plan"]);
					const disallowed = toolCalls.filter((tc) => !planAllowed.has(tc.name));
					if (disallowed.length > 0) {
						await emit({ type: "turn_end", message, toolResults: [] });
						const blocked = disallowed.map((tc) => `\`${tc.name}\``).join(", ");
						const allowList = [...planAllowed].map((n) => `\`${n}\``).join(", ");
						pendingMessages.push({
							role: "user",
							content: [{ type: "text", text: `Mode violation: ${blocked} is not allowed in PLAN mode. Allowed tools: ${allowList}.` }],
							timestamp: ((Date.now() - loopStart) / 1000),
						});
						hasMoreToolCalls = false;
						continue;
					}

					const toolSignature = toolCalls.map((tc: any) => {
						if (!tc || tc.type !== "toolCall") return "invalid";
						const args = tc.arguments as Record<string, unknown> | undefined;
						const p = typeof args?.path === "string" ? args.path : "";
						const fp = typeof args?.file_path === "string" ? args.file_path : "";
						return `${tc.name}:${p || fp}`;
					}).join("|");
					if (toolSignature.length > 0) {
						if (toolSignature === lastToolSignature) repeatedToolSignatureCount++;
						else {
							lastToolSignature = toolSignature;
							repeatedToolSignatureCount = 1;
							lastSignatureMutationCount = successfulMutationCount;
						}
					}

					toolResults.push(
						...(hasPlanToolCall
							? await executePlanModeToolBatchWithHandshake(message, toolCalls)
							: await executeToolCalls(currentContext, message, config, signal, emit, loopStart)),
					);
					for (const result of toolResults) {
						currentContext.messages.push(result);
						newMessages.push(result);
					}

					for (let bi = 0; bi < toolResults.length; bi++) {
						const tr = toolResults[bi];
						const tc = toolCalls[bi];
						if (tr.toolName === "bash" && !tr.isError) {
							const output = tr.content?.map((c: any) => c.text ?? "").join("") ?? "";
							if (output.includes("ConnectionRefusedError") || output.includes("Connection refused") || output.includes("ECONNREFUSED")) {
								pendingMessages.push({ role: "user", content: [{ type: "text", text: "No services available in this environment. Network installs and requests will fail." }], timestamp: ((Date.now() - loopStart) / 1000) });
								break;
							}
							const cmd =
								tc && tc.type === "toolCall" && tc.name === "bash"
									? String((tc.arguments as { command?: string })?.command ?? "")
									: "";
							const haystack = `${cmd}\n${output}`;
							if (/\bnpm\s+(?:i|install|ci)\b/i.test(haystack) || /\bpnpm\s+(?:i|install|add)\b/i.test(haystack) || /\byarn\s+(?:add|install)\b/i.test(haystack)) {
								pendingMessages.push({
									role: "user",
									content: [{ type: "text", text: "Package installs are blocked offline. Skip new installs." }],
									timestamp: ((Date.now() - loopStart) / 1000),
								});
								break;
							}
						}
						if ((tr.toolName === "find" || tr.toolName === "grep") && tr.isError) {
							const errText = tr.content?.map((c: any) => c.text ?? "").join("") ?? "";
							if (errText.includes("fd is not available") || errText.includes("ripgrep") || errText.includes("not available")) {
								const tcFind = toolCalls.find((c: any) => c.type === "toolCall" && c.name === tr.toolName);
								if (tcFind) {
									const args = tcFind.arguments as any;
									let bashCmd = "";
									if (tr.toolName === "find") {
										const pattern = args?.pattern || args?.glob || "*";
										const dir = args?.path || ".";
										bashCmd = `find ${dir} -type f -name "${pattern}" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/dist/*" | head -30`;
									} else {
										const pattern = args?.pattern || "";
										const searchPath = args?.path || ".";
										const glob = args?.glob ? `--include="${args.glob}"` : "";
										bashCmd = `grep -rnl ${glob} --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=dist "${pattern}" ${searchPath} | head -20`;
									}
									pendingMessages.push({
										role: "user",
										content: [{ type: "text", text: `The ${tr.toolName} tool is unavailable. Use bash instead:\n\`\`\`\n${bashCmd}\n\`\`\`\nRun this with \`bash\` now.` }],
										timestamp: ((Date.now() - loopStart) / 1000),
									});
								}
							}
						}
					}

					for (let i = 0; i < toolResults.length; i++) {
						const tr = toolResults[i];
						const tc = toolCalls[i];
						if ((tr.toolName === "read" || tr.toolName === "bash") && !tr.isError && !hasProducedEdit) {
							explorationCount++;
							totalExplorationSteps++;
						}
						if (tr.toolName === "read" && !tr.isError && tc && tc.type === "toolCall" && tc.name === "read") {
							const path = resolvedLoopPathForTool(tc) ?? "";
							if (path) {
								absorbedFiles.add(path);
								// Record every successful PLAN-mode read so the implement
								// phase can surface these paths to the model as candidate
								// context files.
								recordPlanModeReadPath(path);
							}
							const rArgs = tc.arguments as { offset?: number; limit?: number } | undefined;
							const isFullRead = typeof rArgs?.offset === "undefined" && typeof rArgs?.limit === "undefined";
							if (isFullRead && path) {
								addReadPathVariants(pathsAlreadyRead, path);
								await maybeAddSiblingHintAfterRead(path);
								pathReadCounts.set(path, (pathReadCounts.get(path) ?? 0) + 1);
							}
						}
					}
					for (let i = 0; i < toolResults.length; i++) {
						const tr = toolResults[i];
						const tc = toolCalls[i];
						if (!tc || tc.type !== "toolCall" || tc.name !== "plan") continue;
						const planEchoDetails = (tr as any)?.details as { planDraftEcho?: boolean } | undefined;
						if (planEchoDetails?.planDraftEcho === true) {
							continue;
						}
						const planBypassExceeded = (Date.now() - loopStart) >= PLAN_MODE_BYPASS_MS;
						if (tr.isError) {
							const errText = tr.content?.map((c: any) => c?.text ?? "").join("") ?? "unknown plan validation error";
							await emitRolloutMarker("plan_validation_error", {
								error: errText,
							});
							if (planBypassExceeded) {
								await emitRolloutMarker("plan_validation_timeout_bypass", {
									reason: "tool_error",
									error: errText,
								});
							} else {
								pendingMessages.push({
									role: "user",
									content: [{
										type: "text",
										text:
											`PLAN mode remediation: your \`plan\` call failed validation and cannot transition to IMPLEMENT yet.\n` +
											`Stay in PLAN mode and submit a corrected \`plan\`.\n` +
											`Validator error:\n${errText}\n\n` +
											`Fix checklist (do all):\n` +
											`1) Use exact payload keys: \`task_acceptance_criteria\`, \`plans\`, and for each plan item: \`path\`, \`plan\`, \`acceptance_criteria\`, \`is_new_file\`.\n` +
											`2) Ensure \`plans\` is non-empty and under the max file cap.\n` +
											`3) Ensure each \`plans[].acceptance_criteria\` is non-empty and maps to official task criteria.\n` +
											`4) Ensure each \`plans[].plan\` is implement-ready with explicit \`Scope:\`, \`Edits:\`, \`Acceptance:\`, \`Verification:\`.\n` +
											`5) Ensure every \`plans[].path\` is an EXACT verbatim path proven by \`ls\`/\`find\`/\`grep\` output (no guessed paths).\n` +
											`6) Ensure \`is_new_file\` is correct for each path.\n\n` +
											`Then call \`plan\` again (tool call only).`,
									}],
									timestamp: ((Date.now() - loopStart) / 1000),
								});
								continue;
							}
						}
						const planDetails = (tr as any)?.details as {
							allPassed?: boolean;
							uncoveredCriteria?: string[];
							validationResults?: unknown[];
						} | undefined;
						if (!planDetails?.allPassed) {
							const preview = Array.isArray(planDetails?.validationResults)
								? planDetails!.validationResults.map((v: any, idx: number) => {
									const p = typeof v?.path === "string" ? v.path : "(missing path)";
									const status = typeof v?.validation_result === "string" ? v.validation_result : "unknown";
									const sug = Array.isArray(v?.suggested_paths) && v.suggested_paths.length > 0 ? ` suggestions: ${v.suggested_paths.slice(0, 5).join(", ")}` : "";
									return `#${idx + 1} ${p} => ${status}${sug}`;
								}).join("; ")
								: "Plan validation failed.";
							const failedPaths = Array.isArray(planDetails?.validationResults)
								? planDetails!.validationResults.filter((v: any) => String(v?.validation_result ?? "") !== "passed").map((v: any) => (typeof v?.path === "string" ? v.path : "")).filter((p: string) => p.length > 0)
								: [];
							const failedPathNudge = failedPaths.length > 0 ? `Failed planned file paths to verify/fix now: ${failedPaths.slice(0, 10).map((p: string) => `\`${p}\``).join(", ")}.` : "";
							const uncovered = Array.isArray(planDetails?.uncoveredCriteria) && planDetails!.uncoveredCriteria.length > 0
								? `Uncovered task criteria: ${planDetails!.uncoveredCriteria.slice(0, 6).map((c: string) => `"${c}"`).join(" | ")}.`
								: "";
							const strictPathCorrectionInstruction =
								failedPaths.length > 0
									? "Check the suggested filepaths first. If there isn't the right path what you really target, then discover the file again. Don't call plan tool again without wrong path correction."
									: "";
							if (!planBypassExceeded) {
								pendingMessages.push({
									role: "user",
									content: [{
										type: "text",
										text:
											`PLAN mode remediation: plan validation failed. Stay in PLAN mode.\n` +
											`${failedPathNudge ? `${failedPathNudge}\n` : ""}` +
											`${uncovered ? `${uncovered}\n` : ""}` +
											`Detailed validator results:\n${preview}\n\n` +
											`Required correction before next \`plan\` call:\n` +
											`- Correct every failed path using EXACT verbatim discovery-proven paths only (\`ls\`/\`find\`/\`grep\` output), or set \`is_new_file\` appropriately.\n` +
											`${strictPathCorrectionInstruction ? `- ${strictPathCorrectionInstruction}\n` : ""}` +
											`- Ensure criteria coverage is complete and correctly mapped.\n` +
											`- Keep each plan item fully implementable (no ambiguity in \`Edits:\`).\n` +
											`- Re-submit one corrected \`plan\` payload.`,
									}],
									timestamp: ((Date.now() - loopStart) / 1000),
								});
								continue;
							}
							await emitRolloutMarker("plan_validation_timeout_bypass", {
								reason: "details_not_all_passed",
								failed_paths: failedPaths,
							});
						}
						const plans = extractPlanItems(tc.arguments);
						await emitRolloutMarker("plan_extracted", {
							plans,
						});
						if (plans.length === 0 && !planBypassExceeded) {
							pendingMessages.push({
								role: "user",
								content: [{ type: "text", text: "Plan submission was empty or malformed. Stay in PLAN mode and call `plan` with exact JSON format. Plans must be detailed." }],
								timestamp: ((Date.now() - loopStart) / 1000),
							});
							continue;
						}
						const submittedPlanPaths = new Set<string>(plans.map((p) => (typeof p?.path === "string" ? normalizePathForMatch(p.path.trim()) : "")).filter((p) => p.length > 0));
						const missingRequiredInPlan = missingRequiredFromPlan(submittedPlanPaths);
						if (missingRequiredInPlan.length > 0) {
							if (planBypassExceeded) {
								await emitRolloutMarker("plan_validation_timeout_bypass", {
									reason: "missing_required_paths",
									missing_required_in_plan: missingRequiredInPlan,
								});
							} else {
								pendingMessages.push({
									role: "user",
									content: [{ type: "text", text: `Plan coverage incomplete in PLAN mode. Missing required files in \`plan\`: ${missingRequiredInPlan.slice(0, 12).map((f) => `\`${f}\``).join(", ")}. Add them and call \`plan\` again before transition.` }],
									timestamp: ((Date.now() - loopStart) / 1000),
								});
								continue;
							}
						}
						for (const p of plans) {
							if (typeof p?.path !== "string" || p.path.trim().length === 0) continue;
							const np = normalizePathForMatch(p.path.trim());
							plannedFiles.add(np);
							if (typeof p?.plan === "string" && p.plan.trim().length > 0) planByFile.set(np, p.plan.trim());
							if (!foundFiles.includes(np)) foundFiles.push(np);
						}
						{
							const lines: string[] = [];
							const submittedAtMs = Date.now();
							const submittedAtIso = new Date(submittedAtMs).toISOString();
							const elapsedMs = submittedAtMs - loopStart;
							const elapsedSec = (elapsedMs / 1000).toFixed(2);
							lines.push(
								`[plan] submitted ${plans.length} item(s) submitted_at=${submittedAtIso} elapsed_since_loop_start_s=${elapsedSec}`,
							);
							for (let idx = 0; idx < plans.length; idx++) {
								const p: any = plans[idx];
								const pathText = typeof p?.path === "string" ? p.path.trim() : "";
								const isNew = Boolean(p?.is_new_file);
								const acceptanceRefs = Array.isArray(p?.acceptance_criteria)
									? p.acceptance_criteria.filter((c: unknown) => typeof c === "string" && c.trim().length > 0)
									: [];
								const planText = typeof p?.plan === "string" ? p.plan.trim() : "";
								lines.push(`#${idx + 1} path=${pathText} is_new_file=${isNew} submitted_at=${submittedAtIso} elapsed_since_loop_start_s=${elapsedSec}`);
								if (acceptanceRefs.length > 0) {
									lines.push(`acceptance_criteria=${acceptanceRefs.join(" | ")}`);
								}
								lines.push("plan:");
								lines.push(planText || "(empty)");
								lines.push("---");
							}
							process.stdout.write(`${lines.join("\n")}\n`);
						}
						{
							const submittedAtMs = Date.now();
							await emitRolloutMarker("plan_submitted", {
								plan_count: plans.length,
								paths: plans.map((p) => (typeof p?.path === "string" ? normalizePathForMatch(p.path.trim()) : "")).filter((p) => p.length > 0),
								submitted_at: new Date(submittedAtMs).toISOString(),
								submitted_ts_ms: submittedAtMs,
								elapsed_since_loop_start_ms: submittedAtMs - loopStart,
								handshake: "validated_second_plan_only_turn",
							});
							await emitRolloutMarker("plan_handshake_validated", {
								plan_count: plans.length,
							});
						}
						planSubmitted = true;
						planHandshakeAwaitingConsecutivePlanOnly = false;
						executionMode = "implement";
						implementModeStartedAt = Date.now();
						if (planBypassExceeded) {
							pendingMessages.push({
								role: "user",
								content: [{
									type: "text",
									text:
										"PLAN timeout bypass triggered (~150s). Mode is now changed into IMPLEMENT automatically using the latest submitted plan. " +
										"These plans might not be perfect — implement them as-is and proceed file-by-file.",
								}],
								timestamp: ((Date.now() - loopStart) / 1000),
							});
						}
						// Build the ordered plan list that drives the implement-mode for-loop.
						plannedOrder.length = 0;
						for (const p of plans) {
							const path = typeof p?.path === "string" ? p.path.trim() : "";
							const planText = typeof p?.plan === "string" ? p.plan.trim() : "";
							if (path.length > 0) plannedOrder.push({ path, plan: planText });
						}
						currentPlanIndex = 0;
						currentPlanStartedAt = Date.now();
						originalContentByPlanIndex.clear();
						pendingEditdoneConfirmationPlanIndex = null;
						ensurePlanOriginalSnapshot(currentPlanIndex);
						// Freeze the "stable prefix" for implement mode: everything in
						// `currentContext.messages` up to this point (the task,
						// discovery, `plan` tool call + result) will always be preserved
						// by both the within-plan and cross-plan trim helpers.
						implementModeBaseMsgIdx = currentContext.messages.length;
						await emitRolloutMarker("mode_transition", {
							from: "plan",
							to: "implement",
							reason: "plan_submitted",
							implement_base_msg_idx: implementModeBaseMsgIdx,
						});
						if (plannedOrder.length > 0) {
							const firstMsg = buildCurrentPlanInjectMessage();
							if (firstMsg) queueImplementInjectMessage(firstMsg);
						}
					}
				}
			} else if (executionMode === "implement") {
				// Simple per-plan for-loop:
				//   1. When entering implement mode (or after an `editdone`), the plan-
				//      submission block above pushes a message containing the target
				//      file's current full content + plan and asks for edit/write/editdone.
				//   2. Each turn, the model must call `read`, `edit`, `write`, or `editdone`.
				//   3. After `edit`/`write` on the current plan path only, we execute the
				//      tool and re-inject the (possibly updated) file content + plan
				//      with the same "call edit/write/editdone" instruction.
				//   4. After `editdone`, we advance to the next plan (or finish if the
				//      model has called editdone once per plan).
				if (currentPlanIndex >= plannedOrder.length) {
					await emit({ type: "turn_end", message, toolResults: [] });
					await emit({ type: "agent_end", messages: newMessages });
					return;
				}

				const currentPlan = plannedOrder[currentPlanIndex];

				if (!hasMoreToolCalls) {
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push({
						role: "user",
						content: [{
							type: "text",
							text:
								`You must call a tool. For the current plan (\`${currentPlan.path}\`), call \`read\`, \`edit\`, \`write\`, or \`editdone\` now. ` +
								`\`edit\` and \`write\` must target the current plan file: \`${currentPlan.path}\`; plan-mode paths are for \`read\` only.`,
						}],
						timestamp: ((Date.now() - loopStart) / 1000),
					});
					continue;
				}

				// Execute tools; `edit`/`write` are only executed when the path matches the current plan file.
				toolResults.push(
					...(await executeImplementModeToolCalls(
						currentContext,
						message,
						toolCalls,
						currentPlan.path,
						config,
						signal,
						emit,
						loopStart,
					)),
				);
				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}

				// Record successful file mutations for stats only (not used as a guard).
				for (let i = 0; i < toolResults.length; i++) {
					const tr = toolResults[i];
					const tc = toolCalls[i];
					if (!tc || tc.type !== "toolCall") continue;
					if (tr.isError) continue;
					if (tc.name === "edit" || tc.name === "write") {
						const p = resolvedLoopPathForTool(applyRobustWorkspacePathsToToolCall(tc));
						if (p) {
							await recordSuccessfulFileMutation(p);
							refreshNetChangedStateForPath(p);
						}
					}
				}

				// Did the model signal the current plan is done?
				const hasEditdone = toolCalls.some((tc) => tc.name === "editdone");
				// "Twice in a row" semantics: any non-editdone turn resets pending confirmation.
				if (!hasEditdone && pendingEditdoneConfirmationPlanIndex === currentPlanIndex) {
					pendingEditdoneConfirmationPlanIndex = null;
				}
				if (hasEditdone) {
					const firstEditdoneCall = toolCalls.find((tc) => tc.name === "editdone");
					const firstEditdoneArgs = (firstEditdoneCall?.arguments ?? {}) as Record<string, unknown>;
					const evidence =
						typeof firstEditdoneArgs.completedevidence === "string"
							? firstEditdoneArgs.completedevidence
							: "";
					const evidenceDetailed = isDetailedCompletionEvidence(evidence);
					const isSecondConsecutiveDetailedEditdone =
						pendingEditdoneConfirmationPlanIndex === currentPlanIndex && evidenceDetailed;

					if (isSecondConsecutiveDetailedEditdone) {
						const completedPlanIndex = currentPlanIndex;
						pendingEditdoneConfirmationPlanIndex = null;
						currentPlanIndex++;
						currentPlanStartedAt = Date.now();
						if (currentPlanIndex >= plannedOrder.length) {
							await emit({ type: "turn_end", message, toolResults });
							await emit({ type: "agent_end", messages: newMessages });
							return;
						}
						ensurePlanOriginalSnapshot(currentPlanIndex);
						// Drop the just-completed plan's iteration from context so the next
						// plan starts with the stable PLAN-mode prefix + one fresh inject.
						await emit({ type: "turn_end", message, toolResults });
						resetToImplementModeBase("editdone_advance", completedPlanIndex);
						const nextMsg = buildCurrentPlanInjectMessage();
						if (nextMsg) queueImplementInjectMessage(nextMsg);
						continue;
					}

					pendingEditdoneConfirmationPlanIndex = currentPlanIndex;
					await emit({ type: "turn_end", message, toolResults });
					const confirmMsg = buildEditdoneConfirmationInjectMessage(currentPlanIndex, evidence);
					if (confirmMsg) queueImplementInjectMessage(confirmMsg);
					continue;
				}

				let hadEditLineFailure = false;
				for (let i = 0; i < toolResults.length; i++) {
					const tr = toolResults[i];
					const tc = toolCalls[i];
					if (!tc || tc.type !== "toolCall" || tc.name !== "edit" || !tr.isError) continue;
					const errText = (tr.content ?? [])
						.filter((c): c is { type: "text"; text: string } => c.type === "text" && typeof (c as { text?: unknown }).text === "string")
						.map((c) => c.text)
						.join("\n");
					// Wrong-path blocks are a different issue; only nudge line numbers for real edit failures.
					if (errText.includes("Implement mode:") && errText.includes("must target ONLY")) continue;
					hadEditLineFailure = true;
					break;
				}
				if (hadEditLineFailure) {
					pendingMessages.push({
						role: "user",
						content: [{
							type: "text",
							text:
								"Your `edit` tool call failed. You memory is wrong. Refresh your memory with the new content of target file. Re-check `oldText` against the **refreshed** file content. After that, make `edit` tool call payload correctly and call `edit` tool again with correct payload. Do not repeat the same mistake.",
						}],
						timestamp: ((Date.now() - loopStart) / 1000),
					});
				}

				// Per-plan time budget auto-advance: if the current plan has already
				// consumed more than its share of the remaining ultimate budget, skip
				// ahead to the next plan instead of re-injecting the same context and
				// wasting more time on a stuck `edit` loop. This is the key guard that
				// prevents getting stuck on plan N/K while plans N+1..K never run.
				if (currentPlanStartedAt !== null) {
					const planElapsed = Date.now() - currentPlanStartedAt;
					const planBudget = computePlanBudgetMs();
					if (planElapsed >= planBudget) {
						await emitRolloutMarker("plan_budget_exceeded", {
							plan_index: currentPlanIndex,
							plan_path: currentPlan.path,
							elapsed_ms: planElapsed,
							budget_ms: planBudget,
						});
						const abandonedPlanIndex = currentPlanIndex;
						pendingEditdoneConfirmationPlanIndex = null;
						currentPlanIndex++;
						currentPlanStartedAt = Date.now();
						if (currentPlanIndex >= plannedOrder.length) {
							await emit({ type: "turn_end", message, toolResults });
							await emit({ type: "agent_end", messages: newMessages });
							return;
						}
						// Same cross-plan wipe as the editdone path: the next plan
						// starts from the clean PLAN-mode prefix.
						await emit({ type: "turn_end", message, toolResults });
						resetToImplementModeBase("plan_budget_exceeded", abandonedPlanIndex);
						ensurePlanOriginalSnapshot(currentPlanIndex);
						const skipMsg = buildCurrentPlanInjectMessage();
						if (skipMsg) queueImplementInjectMessage(skipMsg);
						continue;
					}
				}

				// After any edit/write (successful or not), re-inject the current file
				// content + plan + instruction. This is the "second step" of the per-plan
				// loop: the agent always follows up a tool call with the refreshed context
				// so the model can either continue editing or call `editdone`.
				//
				// `queueImplementInjectMessage` keeps at most ONE inject live at a
				// time: any prior inject for this plan is removed before the new one
				// is queued, so within-plan prompt growth stays bounded (~1 inject
				// instead of N stacked file dumps).
				const reinjectMsg = buildCurrentPlanInjectMessage();
				if (reinjectMsg) queueImplementInjectMessage(reinjectMsg);
			}
			// Plan-mode-only stall exit. Implement-mode completion is decided by the
			// per-plan for-loop (on the Nth `editdone`), so we never terminate here
			// for implement mode regardless of tool-signature repetition.
			if (
				executionMode === "plan" &&
				hasProducedEdit &&
				repeatedToolSignatureCount >= REPEATED_TOOL_SIGNATURE_LIMIT &&
				successfulMutationCount === lastSignatureMutationCount &&
				planSubmitted &&
				pendingMessages.length === 0
			) {
				await emit({ type: "turn_end", message, toolResults });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			await emit({ type: "turn_end", message, toolResults });

			// Preserve in-loop nudges (e.g. PLAN->IMPLEMENT handoff) and append external steering.
			const steeringMessages = (await config.getSteeringMessages?.()) || [];
			if (steeringMessages.length > 0) {
				pendingMessages.push(...steeringMessages);
			}
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			pendingMessages = followUpMessages;
			continue;
		}
		// Hard cutoff for implementation flow: stop after IMPLEMENT_VERIFY_MAX_MS.
		if (
			executionMode === "implement" &&
			implementModeStartedAt !== null &&
			(Date.now() - implementModeStartedAt) >= IMPLEMENT_VERIFY_MAX_MS
		) {
			break;
		}

		if (executionMode === "plan") {
			pendingMessages = [{
				role: "user",
				content: [{
					type: "text",
					text: "Cannot stop in PLAN mode. Continue exploration and submit final file-by-file plan via `plan` tool.",
				}],
				timestamp: ((Date.now() - loopStart) / 1000),
			}];
			continue;
		}

		if (executionMode === "implement") {
			if (!planSubmitted) {
				await emitRolloutMarker("mode_transition", {
					from: "implement",
					to: "plan",
					reason: "invariant_correction",
				});
				executionMode = "plan";
				pendingMessages = [{
					role: "user",
					content: [{
						type: "text",
						text:
							"Invariant correction: implement mode requires a successful `plan` tool call first. Return to PLAN mode and submit a valid plan.",
					}],
					timestamp: ((Date.now() - loopStart) / 1000),
				}];
				continue;
			}
			// If the per-plan for-loop is not yet exhausted, keep going: re-inject the
			// current plan's context so the model must call edit/write/editdone again.
			// Route through the trim-aware queue helper so we do not stack another
			// file-content dump on top of any stale inject still in context.
			if (currentPlanIndex < plannedOrder.length) {
				const cur = buildCurrentPlanInjectMessage();
				if (cur) queueImplementInjectMessage(cur);
				continue;
			}
			// All plans signalled done — exit.
			break;
		}

		// No more messages, exit
		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
type StreamOverrideOptions = {
	toolChoice?: "auto" | "none" | "required" | { type: "function"; function: { name: string } };
};

async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
	streamOverrides?: StreamOverrideOptions,
	/** When set (e.g. τ dual-prompt mode), overrides \`context.systemPrompt\` for the LLM request only. */
	llmSystemPrompt?: string,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	// Build LLM context
	const llmContext: Context = {
		systemPrompt: llmSystemPrompt ?? context.systemPrompt,
		messages: llmMessages,
		tools: context.tools,
	};

	const streamFunction = streamFn || streamSimple;

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
		...(streamOverrides?.toolChoice !== undefined ? { toolChoice: streamOverrides.toolChoice } : {}),
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_start":
			case "text_delta":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					await emit({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: { ...finalMessage } });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	const finalMessage = await response.result();
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: { ...finalMessage } });
	}
	await emit({ type: "message_end", message: finalMessage });
	return finalMessage;
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	loopStart: number,
): Promise<ToolResultMessage[]> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	if (config.toolExecution === "sequential") {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit, loopStart);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit, loopStart);
}

/** Same normalization as in-loop `normalizePathForMatch` for implement-mode path checks. */
function normalizeImplementPlanPath(p: string): string {
	return p.replace(/^\.\//, "");
}

/**
 * Implement mode: run tool calls sequentially and ensure `edit` / `write` only touch
 * the current planned file. Other paths from plan mode exist for `read` only.
 */
async function executeImplementModeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	plannedTargetPath: string,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	loopStart: number,
): Promise<ToolResultMessage[]> {
	const planNorm = normalizeImplementPlanPath(plannedTargetPath.trim());
	const results: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		const tc = applyRobustWorkspacePathsToToolCall(toolCall);
		await emit({
			type: "tool_execution_start",
			toolCallId: tc.id,
			toolName: tc.name,
			args: tc.arguments,
		});

		if (tc.name === "edit" || tc.name === "write") {
			const resolved = resolvedLoopPathForTool(tc);
			const toolNorm = resolved ? normalizeImplementPlanPath(resolved) : "";
			if (!resolved || toolNorm !== planNorm) {
				const wrong = resolved ?? rawPathFromToolArguments(tc.arguments)?.trim() ?? "(missing path)";
				results.push(
					await emitToolCallOutcome(
						tc,
						createErrorToolResult(
							`Implement mode: \`${tc.name}\` must target ONLY the current planned file \`${plannedTargetPath}\`. ` +
								`You targeted \`${wrong}\`. ` +
								`Paths recorded during plan mode are for \`read\` only — call \`edit\` or \`write\` with path exactly \`${plannedTargetPath}\`.`,
						),
						true,
						emit,
						loopStart,
					),
				);
				continue;
			}
		}

		const preparation = await prepareToolCall(currentContext, assistantMessage, tc, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(tc, preparation.result, preparation.isError, emit, loopStart));
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			results.push(
				await finalizeExecutedToolCall(
					currentContext,
					assistantMessage,
					preparation,
					executed,
					config,
					signal,
					emit,
					loopStart,
				),
			);
		}
	}

	return results;
}

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	loopStart: number,
): Promise<ToolResultMessage[]> {
	const results: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit, loopStart));
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			results.push(
				await finalizeExecutedToolCall(
					currentContext,
					assistantMessage,
					preparation,
					executed,
					config,
					signal,
					emit,
					loopStart,
				),
			);
		}
	}

	return results;
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	loopStart: number,
): Promise<ToolResultMessage[]> {
	const results: ToolResultMessage[] = [];
	const runnableCalls: PreparedToolCall[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit, loopStart));
		} else {
			runnableCalls.push(preparation);
		}
	}

	const runningCalls = runnableCalls.map((prepared) => ({
		prepared,
		execution: executePreparedToolCall(prepared, signal, emit),
	}));

	for (const running of runningCalls) {
		const executed = await running.execution;
		results.push(
			await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				running.prepared,
				executed,
				config,
				signal,
				emit,
				loopStart,
			),
		);
	}

	return results;
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const pathResolvedToolCall = applyRobustWorkspacePathsToToolCall(preparedToolCall);
		const validatedArgs = validateToolArguments(tool, pathResolvedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (beforeResult?.block) {
				return {
					kind: "immediate",
					result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
					isError: true,
				};
			}
		}
		return {
			kind: "prepared",
			toolCall: { ...toolCall, arguments: validatedArgs as Record<string, any> },
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	loopStart: number,
): Promise<ToolResultMessage> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		const afterResult = await config.afterToolCall(
			{
				assistantMessage,
				toolCall: prepared.toolCall,
				args: prepared.args,
				result,
				isError,
				context: currentContext,
			},
			signal,
		);
		if (afterResult) {
			result = {
				content: afterResult.content ?? result.content,
				details: afterResult.details ?? result.details,
			};
			isError = afterResult.isError ?? isError;
		}
	}

	return await emitToolCallOutcome(prepared.toolCall, result, isError, emit, loopStart);
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolCallOutcome(
	toolCall: AgentToolCall,
	result: AgentToolResult<any>,
	isError: boolean,
	emit: AgentEventSink,
	loopStart: number
): Promise<ToolResultMessage> {
	await emit({
		type: "tool_execution_end",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		result,
		isError,
	});

	const toolResultMessage: ToolResultMessage = {
		role: "toolResult",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		content: result.content,
		details: result.details,
		isError,
		timestamp: ((Date.now() - loopStart) / 1000),
	};

	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
	return toolResultMessage;
}
