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

/** Large cap so typical repo files are included in full; extreme files still truncate to protect memory. */
const GEMINI_EDIT_RECOVERY_MAX_CHARS = 8_000_000;

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

function fenceLangFromPath(filePath: string): string {
	const ext = extname(filePath).slice(1).toLowerCase();
	const m: Record<string, string> = {
		ts: "typescript",
		tsx: "tsx",
		js: "javascript",
		jsx: "javascript",
		py: "python",
		rs: "rust",
		go: "go",
		cs: "csharp",
		json: "json",
		html: "html",
		css: "css",
		md: "markdown",
		sh: "bash",
		yml: "yaml",
		yaml: "yaml",
	};
	return m[ext] ?? (ext || "text");
}

const GEMINI_OLDTEXT_RECOVERY_MAX_CHARS = 500_000;

/** Full `oldText` blocks for Gemini recovery (wrong-anchor debugging); cap avoids pathological payloads. */
function extractOldTextForGeminiRecovery(toolCall: AgentToolCall): string {
	const args = toolCall.arguments as { edits?: Array<{ oldText?: string }>; oldText?: string } | undefined;
	if (!args) return "";
	const chunks: string[] = [];
	if (typeof args.oldText === "string" && args.oldText.length > 0) {
		chunks.push(args.oldText);
	}
	if (Array.isArray(args.edits)) {
		for (let i = 0; i < args.edits.length; i++) {
			const ot = args.edits[i]?.oldText;
			if (typeof ot === "string" && ot.length > 0) {
				chunks.push(args.edits!.length > 1 ? `--- edits[${i}].oldText ---\n${ot}` : ot);
			}
		}
	}
	let s = chunks.join("\n\n");
	if (s.length > GEMINI_OLDTEXT_RECOVERY_MAX_CHARS) {
		s = `${s.slice(0, GEMINI_OLDTEXT_RECOVERY_MAX_CHARS)}…`;
	}
	return s;
}

/**
 * Rich recovery text for Gemini after a failed `edit`: actual file bytes (when present) or
 * concrete discovery commands when the path is wrong/missing.
 */
function buildGeminiEditFailureRecoveryMessage(
	targetPath: string,
	errText: string,
	toolCall: AgentToolCall,
	foundFiles: string[],
): string {
	const pathForDisk = resolveWorkspaceToolPathString(targetPath, { basenameFallback: true });
	const { abs, ok } = safeResolvePathUnderCwd(pathForDisk);
	const base = basename(pathForDisk);
	const lines: string[] = [];

	lines.push(`## Edit failed for \`${targetPath}\`${pathForDisk !== targetPath ? ` (resolved to \`${pathForDisk}\`)` : ""}`);
	lines.push("");
	lines.push("**Tool error (verbatim):**");
	lines.push("```text");
	lines.push(escapeMarkdownFences(errText.trim() || "(empty error message)"));
	lines.push("```");
	lines.push("");

	const oldTextDump = extractOldTextForGeminiRecovery(toolCall);
	if (oldTextDump) {
		lines.push("**Your `oldText` from the failed tool call — compare byte-for-byte with the file contents below:**");
		lines.push("```text");
		lines.push(escapeMarkdownFences(oldTextDump));
		lines.push("```");
		lines.push("");
	}

	if (!ok) {
		lines.push(`**Path safety:** resolved path would leave the workspace cwd (\`${process.cwd()}\`) — fix \`..\` or use a repo-relative path.`);
		lines.push("");
		lines.push(`**Discover a safe path:** use \`find\` / \`grep\` from the repo root (see examples below).`);
		lines.push("");
		lines.push(`**Example bash:**`);
		lines.push(`\`\`\`bash`);
		lines.push(`find . -type f -name '*${base}*' -not -path '*/node_modules/*' -not -path '*/.git/*' | head -20`);
		lines.push(`\`\`\``);
		return lines.join("\n");
	}

	if (!existsSync(abs)) {
		lines.push(`**On disk:** \`${abs}\` does **not** exist (cwd=\`${process.cwd()}\`).`);
		lines.push("");
		lines.push("**Recover the real path:**");
		lines.push(`1. Run \`find\` with pattern \`*${base}*\` (or the correct filename) from the repo root.`);
		lines.push(`2. Or \`grep\` for a string that only appears in that file.`);
		lines.push(`3. Use the path returned by the tool — copy it exactly into the next \`edit\` / \`read\`.`);
		const hints = foundFiles.filter(
			(f) =>
				f.replace(/^\.\//, "").endsWith(base) ||
				f.replace(/^\.\//, "").includes(base) ||
				base.includes(basename(f)),
		);
		if (hints.length > 0) {
			lines.push("");
			lines.push("**Candidate paths already discovered in this session (try one):**");
			for (const h of hints.slice(0, 8)) {
				lines.push(`- \`${h}\``);
			}
		}
		lines.push("");
		lines.push("**Example bash (adapt the name):**");
		lines.push(`\`\`\`bash`);
		lines.push(`find . -type f -name '*${base}*' -not -path '*/node_modules/*' -not -path '*/.git/*' | head -20`);
		lines.push(`\`\`\``);
	} else {
		try {
			const st = statSync(abs);
			if (!st.isFile()) {
				lines.push(`**On disk:** \`${abs}\` exists but is not a regular file.`);
			} else {
				const buf = readFileSync(abs);
				const probeLen = Math.min(buf.length, 16_384);
				if (buf.subarray(0, probeLen).includes(0)) {
					lines.push(
						`**Binary or non-UTF8 file** at \`${abs}\` — inline snapshot skipped. Call \`read\` on \`${pathForDisk}\` (or use \`bash\` / \`xxd\` / \`file\` if you need raw inspection).`,
					);
				} else {
					let body = buf.toString("utf8");
					let truncated = false;
					if (body.length > GEMINI_EDIT_RECOVERY_MAX_CHARS) {
						body = body.slice(0, GEMINI_EDIT_RECOVERY_MAX_CHARS);
						truncated = true;
					}
					const lang = fenceLangFromPath(abs);
					lines.push(`**Current full file at \`${pathForDisk}\`** (resolved: \`${abs}\`)${truncated ? ` — **truncated** to ${GEMINI_EDIT_RECOVERY_MAX_CHARS} chars (file is larger). Re-\`read\` with offset/limit if needed.` : ""}:`);
					lines.push(`\`\`\`${lang}`);
					lines.push(escapeMarkdownFences(body));
					lines.push("```");
					lines.push("");
					lines.push(
						"Re-issue `edit` using **exact** `oldText` copied from the file above (whitespace, quotes, line endings). If you only need one line changed, anchor with 2–3 surrounding lines.",
					);
				}
			}
		} catch (e) {
			lines.push(`**Could not read file:** ${e instanceof Error ? e.message : String(e)}`);
		}
	}

	return lines.join("\n");
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
	let implementReadOnlyRequiredTurns = 0;
	const pathsAlreadyRead = new Set<string>();
	const pathReadCounts = new Map<string, number>();
	let lastRereadNudgeAt = 0;
	const editedPaths = new Set<string>();
	const pathEditCounts = new Map<string, number>();
	let consecutiveEditsOnSameFile = 0;
	let lastEditedFile = "";

	let executionMode: "plan" | "implement" = "plan";
	let planSubmitted = false;
	let confirmPhaseStarted = false;
	let confirmPhaseDone = false;
	let confirmCurrentPath: string | null = null;
	const confirmPlanList = new Set<string>();
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
			timestamp: Date.now(),
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
	const needsDeeperPlannedImplementation = (): string[] => {
		const needsMore: string[] = [];
		for (const pf of plannedFiles) {
			const normPf = normalizePathForMatch(pf);
			const planText = planByFile.get(normPf) ?? "";
			if (!planText) continue;
			const backtickRefs = (planText.match(/`[^`]+`/g) ?? []).length;
			const hasDetailedEdits = planText.length >= 500 || backtickRefs >= 6;
			if (!hasDetailedEdits) continue;
			const editCount = pathEditCounts.get(normPf) ?? 0;
			if (editCount < 2) needsMore.push(pf);
		}
		return needsMore;
	};
	const isPathEdited = (path: string): boolean => {
		const n = normalizePathForMatch(path);
		return editedPaths.has(n) || editedPaths.has("./" + n) || editedPaths.has(path);
	};
	const buildConfirmPrompt = (path: string): AgentMessage => {
		const normPath = normalizePathForMatch(path);
		const planText = planByFile.get(normPath) ?? "";
		let fileContent = "";
		try {
			const { abs, ok } = safeResolvePathUnderCwd(normPath);
			if (ok && existsSync(abs)) fileContent = readFileSync(abs, "utf8");
		} catch {
			fileContent = "";
		}
		const body = fileContent.length > 0 ? fileContent : "(file content unavailable)";
		return {
			role: "user",
			content: [{
				type: "text",
				text:
					`CONFIRMPLAN CHECK for \`${normPath}\`.\n` +
					`Plan for this file:\n---\n${planText || "(missing plan text)"}\n---\n\n` +
					`FULL edited file content:\n---\n${escapeMarkdownFences(body)}\n---\n\n` +
					`Reply rules:\n` +
					`- If perfect: reply exactly \`PERFECT\` (no tool call).\n` +
					`- If imperfect: reply \`IMPERFECT\` and include an \`edit\` or \`write\` tool call for this same file in the same turn.\n` +
					`- Do not switch files during this confirm step.`,
			}],
			timestamp: Date.now(),
		};
	};
	const queueNextConfirmPrompt = (): boolean => {
		if (confirmPlanList.size === 0) {
			confirmCurrentPath = null;
			confirmPhaseDone = true;
			return false;
		}
		confirmCurrentPath = [...confirmPlanList][0] ?? null;
		if (!confirmCurrentPath) {
			confirmPhaseDone = true;
			return false;
		}
		pendingMessages.push(buildConfirmPrompt(confirmCurrentPath));
		return true;
	};
	const PLAN_MODE_MAX_MS = 100_000;
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
	/** Hard cap below common evaluator timeout (~600s) so landed edits are emitted before external cutoff. */
	const ABSOLUTE_SESSION_CAP_MS = 500_000;
	/** If identical tool-call signatures repeat with no new successful mutations, force-stop to preserve diff. */
	const REPEATED_TOOL_SIGNATURE_LIMIT = 3;
	let multiFileHintSent = false;
	let reviewPassDone = false;
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
			timestamp: Date.now(),
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
					timestamp: Date.now(),
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
	const maybeAddBreadthHintAfterRead = (readPath: string): void => {
		if (executionMode !== "plan") return;
		const normRead = readPath.replace(/^\.\//, "");
		const uneditedTargets = foundFiles.filter((f: string) => {
			const nf = f.replace(/^\.\//, "");
			return !editedPaths.has(f) && !editedPaths.has(nf) && !editedPaths.has("./" + nf);
		});
		if (uneditedTargets.length === 0) return;
		const readEditCount = pathEditCounts.get(normRead) ?? 0;
		const repeatedReadSameFile = (pathReadCounts.get(readPath) ?? 0) >= 2;
		const breadthHintText =
			repeatedReadSameFile || readEditCount >= 2
				? `PLAN breadth hint: stop focusing on \`${normRead}\`. ${uneditedTargets.length} target file(s) are still untouched: ${uneditedTargets.slice(0, 6).map((f: string) => `\`${f}\``).join(", ")}. Read the next highest-signal uninspected file.`
				: `PLAN breadth hint: ${uneditedTargets.length} candidate file(s) remain: ${uneditedTargets.slice(0, 6).map((f: string) => `\`${f}\``).join(", ")}. Continue discovery breadth-first before final \`plan\`.`;
		pendingMessages.push({
			role: "user",
			content: [{ type: "text", text: breadthHintText }],
			timestamp: Date.now(),
		});
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
				timestamp: Date.now(),
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
			timestamp: Date.now(),
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

			// Hard session cap must run every iteration — many `continue` paths skip the tool tail,
			// so a bottom-only check lets the loop run until an external harness timeout.
			if ((Date.now() - loopStart) >= ABSOLUTE_SESSION_CAP_MS) {
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

			const planBudgetExceeded =
				executionMode === "plan" && !planSubmitted && (Date.now() - loopStart) >= PLAN_MODE_MAX_MS;
			const implementNeedsProgress =
				executionMode === "implement" &&
				(
					missingPlannedFiles().length > 0 ||
					needsDeeperPlannedImplementation().length > 0
				);
			const forceImplementMutationChoice =
				executionMode === "implement" &&
				implementNeedsProgress &&
				implementReadOnlyRequiredTurns >= IMPLEMENT_READ_ONLY_REQUIRED_TURN_LIMIT;
			// During CONFIRMPLAN the model is expected to reply with `PERFECT` as plain text
			// (no tool) or `IMPERFECT` + edit/write. "required"/forced edit would block the
			// PERFECT path and trap the loop until the hard cutoff. Relax tool choice here.
			const isConfirmTurn =
				executionMode === "implement" && confirmPhaseStarted && confirmCurrentPath !== null;

			const message = await streamAssistantResponse(
				currentContext,
				config,
				signal,
				emit,
				streamFn,
				{
					// PLAN mode must never burn turns on prose-only replies; Gemini often does that unless forced.
					// After the PLAN wall clock budget, force a real `plan` tool call so discovery-only loops cannot stall.
					toolChoice:
						executionMode === "plan" && !planSubmitted
							? planBudgetExceeded
								? ({ type: "function", function: { name: "plan" } } as const)
								: ("required" as const)
							: executionMode === "implement"
								? (
									isConfirmTurn
										? ("auto" as const)
										: forceImplementMutationChoice
											? ({ type: "function", function: { name: "edit" } } as const)
											: ("required" as const)
								)
								: implementNeedsProgress
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
						timestamp: Date.now(),
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
				if (hasPlanToolCall && expectedFiles.length > 0 && !planBudgetExceeded) {
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
									.join(", ")}.`,
							}],
							timestamp: Date.now(),
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
							timestamp: Date.now(),
						});
						hasMoreToolCalls = false;
						continue;
					}
					if (hasNetworkProbe) {
						await emit({ type: "turn_end", message, toolResults: [] });
						pendingMessages.push({
							role: "user",
							content: [{ type: "text", text: "PLAN mode forbids network/probe bash commands (they often fail with ConnectionRefusedError in eval sandboxes). Use only local repository discovery tools and then call `plan` with exact JSON format." }],
							timestamp: Date.now(),
						});
						hasMoreToolCalls = false;
						continue;
					}
				}
				if ((Date.now() - loopStart) >= PLAN_MODE_MAX_MS && pendingMessages.length === 0 && !hasPlanToolCall) {
					await emit({ type: "turn_end", message, toolResults: [] });
					pendingMessages.push({
						role: "user",
						content: [{ type: "text", text: "Plan mode timeout reached (~100s). Stop exploration now. Do one final coverage check (each criterion -> file) and call only the `plan` tool with detailed JSON plans." }],
						timestamp: Date.now(),
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
						timestamp: Date.now(),
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
							timestamp: Date.now(),
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

					toolResults.push(...(await executeToolCalls(currentContext, message, config, signal, emit)));
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
								pendingMessages.push({ role: "user", content: [{ type: "text", text: "No services available in this environment. Network installs and requests will fail. Proceed with `read`, `edit`, and `write` only - avoid `npm install` unless unavoidable." }], timestamp: Date.now() });
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
									content: [{ type: "text", text: "Package installs are slow and often blocked offline. Prefer `edit`/`write` using the repo's existing stack; skip new installs unless the task explicitly names a dependency." }],
									timestamp: Date.now(),
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
										timestamp: Date.now(),
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
							if (path) absorbedFiles.add(path);
							const rArgs = tc.arguments as { offset?: number; limit?: number } | undefined;
							const isFullRead = typeof rArgs?.offset === "undefined" && typeof rArgs?.limit === "undefined";
							if (isFullRead && path) {
								addReadPathVariants(pathsAlreadyRead, path);
								maybeAddBreadthHintAfterRead(path);
								await maybeAddSiblingHintAfterRead(path);
								pathReadCounts.set(path, (pathReadCounts.get(path) ?? 0) + 1);
							}
						}
					}
					for (let i = 0; i < toolResults.length; i++) {
						const tr = toolResults[i];
						const tc = toolCalls[i];
						if (!tc || tc.type !== "toolCall" || tc.name !== "plan") continue;
						const rawPlanArgs = tc.arguments ?? {};
						await emitRolloutMarker("plan_tool_args", {
							args: rawPlanArgs,
						});
						const planTimeoutExceeded = (Date.now() - loopStart) >= PLAN_MODE_MAX_MS;
						if (tr.isError) {
							const errText = tr.content?.map((c: any) => c?.text ?? "").join("") ?? "unknown plan validation error";
							await emitRolloutMarker("plan_validation_error", {
								error: errText,
							});
							pendingMessages.push({
								role: "user",
								content: [{ type: "text", text: planTimeoutExceeded ? `PLAN timeout reached, but submitted plan is still invalid: ${errText}. Re-submit \`plan\` immediately with corrected paths/structure. Keep it concise but valid.` : `Plan submission failed: ${errText}. Fix and call \`plan\` again.` }],
								timestamp: Date.now(),
							});
							continue;
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
							pendingMessages.push({
								role: "user",
								content: [{ type: "text", text: `Plan validation failed. Stay in PLAN mode. ${failedPathNudge} ${uncovered} Fix invalid path verification and criteria coverage, then call \`plan\` again. ${preview}` }],
								timestamp: Date.now(),
							});
							continue;
						}
						const plans = extractPlanItems(tc.arguments);
						await emitRolloutMarker("plan_extracted", {
							plans,
						});
						if (plans.length === 0 && !planTimeoutExceeded) {
							pendingMessages.push({
								role: "user",
								content: [{ type: "text", text: "Plan submission was empty or malformed. Stay in PLAN mode and call `plan` with exact JSON format. Plans must be detailed." }],
								timestamp: Date.now(),
							});
							continue;
						}
						const submittedPlanPaths = new Set<string>(plans.map((p) => (typeof p?.path === "string" ? normalizePathForMatch(p.path.trim()) : "")).filter((p) => p.length > 0));
						const missingRequiredInPlan = missingRequiredFromPlan(submittedPlanPaths);
						if (missingRequiredInPlan.length > 0) {
							pendingMessages.push({
								role: "user",
								content: [{ type: "text", text: `Plan coverage incomplete in PLAN mode. Missing required files in \`plan\`: ${missingRequiredInPlan.slice(0, 12).map((f) => `\`${f}\``).join(", ")}. Add them and call \`plan\` again before transition.` }],
								timestamp: Date.now(),
							});
							continue;
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
								`[plan] submitted ${plans.length} item(s) submitted_at=${submittedAtIso} elapsed_since_loop_start_ms=${elapsedMs} submitted_ts_ms=${submittedAtMs} elapsed_since_loop_start_s=${elapsedSec}`,
							);
							for (let idx = 0; idx < plans.length; idx++) {
								const p: any = plans[idx];
								const pathText = typeof p?.path === "string" ? p.path.trim() : "";
								const isNew = Boolean(p?.is_new_file);
								const acceptanceRefs = Array.isArray(p?.acceptance_criteria)
									? p.acceptance_criteria.filter((c: unknown) => typeof c === "string" && c.trim().length > 0)
									: [];
								const planText = typeof p?.plan === "string" ? p.plan.trim() : "";
								lines.push(`#${idx + 1} path=${pathText} is_new_file=${isNew} submitted_at=${submittedAtIso} elapsed_since_loop_start_ms=${elapsedMs}`);
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
							});
						}
						planSubmitted = true;
						executionMode = "implement";
						implementModeStartedAt = Date.now();
						confirmPhaseStarted = false;
						confirmPhaseDone = false;
						confirmCurrentPath = null;
						confirmPlanList.clear();
						await emitRolloutMarker("mode_transition", { from: "plan", to: "implement", reason: "plan_submitted" });
						pendingMessages.push({
							role: "user",
							content: [{ type: "text", text: `Switched to IMPLEMENT mode. Implement all planned files with minimal, style-matched edits. Planned files: ${[...plannedFiles].slice(0, 100).map((p) => `\`${p}\``).join(", ")}. Call \`read\` first (full file) before \`edit\`/\`write\` on that file.` }],
							timestamp: Date.now(),
						});
					}

					const now = Date.now();
					if (now - lastRereadNudgeAt >= 5_000 && pendingMessages.length === 0) {
						for (const [rp, cnt] of pathReadCounts) {
							if (cnt < 3) continue;
							lastRereadNudgeAt = now;
							const normRp = rp.replace(/^\.\//, "");
							const others = foundFiles.filter((f: string) => {
								const normF = f.replace(/^\.\//, "");
								return normF !== normRp && !editedPaths.has(f) && !editedPaths.has(normF) && !editedPaths.has("./" + normF);
							});
							pendingMessages.push({
								role: "user",
								content: [{ type: "text", text: `You have read \`${rp}\` ${cnt} times - stop re-reading it. ${others.length > 0 ? `Move to a file you have not edited yet: ${others.slice(0, 5).map((f: string) => `\`${f}\``).join(", ")}.` : "Apply \`edit\` or \`write\` on a different file or stop."}` }],
								timestamp: Date.now(),
							});
							break;
						}
					}
					const dynamicExploreCeiling = Math.max(3, Math.min(foundFiles.length + 1, 6));
					if (!hasProducedEdit && explorationCount >= dynamicExploreCeiling && pendingMessages.length === 0) {
						pendingMessages.push({
							role: "user",
							content: [{ type: "text", text: `Context gathered (${explorationCount} reads/bashes). Apply your first file change (\`edit\` or \`write\`) to the highest-priority target now. A partial patch always outscores an empty diff.` }],
							timestamp: Date.now(),
						});
						explorationCount = 0;
					}
				}
			} else if (executionMode === "implement") {
				if (confirmPhaseStarted && confirmCurrentPath) {
					const assistantText = message.content
						.filter((c: any) => c?.type === "text" && typeof c?.text === "string")
						.map((c: any) => c.text)
						.join("\n");
					const hasPerfect = /\bPERFECT\b/.test(assistantText);
					const hasImperfect = /\bIMPERFECT\b/.test(assistantText);
					const hasFixCall = toolCalls.some((tc) => tc.name === "edit" || tc.name === "write");
					if (hasPerfect && !hasFixCall) {
						confirmPlanList.delete(confirmCurrentPath);
						confirmCurrentPath = null;
						await emit({ type: "turn_end", message, toolResults: [] });
						if (!queueNextConfirmPrompt()) {
							pendingMessages.push({
								role: "user",
								content: [{ type: "text", text: "All planned files confirmed PERFECT." }],
								timestamp: Date.now(),
							});
						}
						hasMoreToolCalls = false;
						continue;
					}
					if (hasImperfect && !hasFixCall) {
						await emit({ type: "turn_end", message, toolResults: [] });
						pendingMessages.push({
							role: "user",
							content: [{
								type: "text",
								text: `CONFIRMPLAN for \`${confirmCurrentPath}\`: you returned IMPERFECT without a fix tool call. Submit \`edit\` or \`write\` for this file now.`,
							}],
							timestamp: Date.now(),
						});
						hasMoreToolCalls = false;
						continue;
					}
					if (hasFixCall) {
						const badTarget = toolCalls.some((tc) => {
							if (tc.name !== "edit" && tc.name !== "write") return false;
							const p = resolvedLoopPathForTool(tc);
							if (!p) return false;
							const np = normalizePathForMatch(p);
							const nc = normalizePathForMatch(confirmCurrentPath!);
							return !(np === nc || np.endsWith("/" + nc) || nc.endsWith("/" + np));
						});
						if (badTarget) {
							await emit({ type: "turn_end", message, toolResults: [] });
							pendingMessages.push({
								role: "user",
								content: [{
									type: "text",
									text: `CONFIRMPLAN fix calls must target only \`${confirmCurrentPath}\`. Retry with \`edit\`/\`write\` for that file.`,
								}],
								timestamp: Date.now(),
							});
							hasMoreToolCalls = false;
							continue;
						}
					}
					if (!hasMoreToolCalls && !hasPerfect && !hasImperfect) {
						await emit({ type: "turn_end", message, toolResults: [] });
						pendingMessages.push({
							role: "user",
							content: [{
								type: "text",
								text: `CONFIRMPLAN for \`${confirmCurrentPath}\`: reply with \`PERFECT\` or \`IMPERFECT\` (+ fix tool call).`,
							}],
							timestamp: Date.now(),
						});
						continue;
					}
				}
				const isImplementReadOnlyTurn =
					implementNeedsProgress &&
					hasMoreToolCalls &&
					toolCalls.every((tc) => tc.name === "read");
				if (isImplementReadOnlyTurn) {
					implementReadOnlyRequiredTurns++;
				} else if (hasMoreToolCalls || !implementNeedsProgress) {
					implementReadOnlyRequiredTurns = 0;
				}
				if (
					isImplementReadOnlyTurn &&
					implementReadOnlyRequiredTurns >= IMPLEMENT_READ_ONLY_REQUIRED_TURN_LIMIT &&
					pendingMessages.length === 0
				) {
					const remainingPlanned = missingPlannedFiles();
					const shallowPlanned = needsDeeperPlannedImplementation();
					const nextTarget = remainingPlanned[0] ?? shallowPlanned[0] ?? [...plannedFiles][0] ?? "";
					pendingMessages.push({
						role: "user",
						content: [{
							type: "text",
							text: `Implement-mode anti-stall: ${implementReadOnlyRequiredTurns} consecutive read-only turns while planned work remains. Stop read-only turns and apply \`edit\` or \`write\` now${nextTarget ? ` on \`${nextTarget}\`` : ""}.`,
						}],
						timestamp: Date.now(),
					});
				}
				if (!hasMoreToolCalls) {
					await emit({ type: "turn_end", message, toolResults: [] });
					// Only require a tool call when planned implementation work is still pending.
					// If all planned files are already edited and the confirm phase has not started,
					// let flow fall through to the bottom-of-loop exit gate so the confirm phase
					// can be initiated. Otherwise we would spin in a "requires tool calls" loop
					// and the confirm phase would never start.
					// If the confirm phase is already running, the CONFIRMPLAN reply handler above
					// has already pushed its own reminder and `continue`d, so this branch only
					// runs when confirm is not active.
					if (implementNeedsProgress && !confirmPhaseStarted) {
						pendingMessages.push({
							role: "user",
							content: [{
								type: "text",
								text: "IMPLEMENT mode requires tool calls every turn. Call `read`, `edit`, or `write` now for planned files only.",
							}],
							timestamp: Date.now(),
						});
						continue;
					}
					// No planned work left AND confirm not started -> allow fall-through so the
					// exit-gate at the bottom of the loop can start the confirm phase on the next
					// iteration.
				}
				if (hasMoreToolCalls) {
					const implementAllowed = new Set(["read", "edit", "write"]);
					const disallowed = toolCalls.filter((tc) => !implementAllowed.has(tc.name));
					if (disallowed.length > 0) {
						await emit({ type: "turn_end", message, toolResults: [] });
						const blocked = disallowed.map((tc) => `\`${tc.name}\``).join(", ");
						const allowList = [...implementAllowed].map((n) => `\`${n}\``).join(", ");
						pendingMessages.push({
							role: "user",
							content: [{ type: "text", text: `Mode violation: ${blocked} is not allowed in IMPLEMENT mode. Allowed tools: ${allowList}.` }],
							timestamp: Date.now(),
						});
						hasMoreToolCalls = false;
						continue;
					}
					for (const tc of toolCalls) {
						if (!tc || tc.type !== "toolCall" || tc.name !== "edit") continue;
						const tPath = resolvedLoopPathForTool(tc) ?? "";
						const args = (tc.arguments as any) ?? {};
						const edits = Array.isArray(args.edits) ? args.edits : [];
						if (edits.length > 6) {
							pendingMessages.push({ role: "user", content: [{ type: "text", text: `Edit call on \`${tPath || "unknown file"}\` has ${edits.length} blocks. Split into smaller calls (max 6 blocks) to reduce mismatch and over-edit risk.` }], timestamp: Date.now() });
							hasMoreToolCalls = false;
							break;
						}
					}
					if (!hasMoreToolCalls) {
						await emit({ type: "turn_end", message, toolResults: [] });
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

					toolResults.push(...(await executeToolCalls(currentContext, message, config, signal, emit)));
					for (const result of toolResults) {
						currentContext.messages.push(result);
						newMessages.push(result);
					}

					for (let i = 0; i < toolResults.length; i++) {
						const tr = toolResults[i];
						const tc = toolCalls[i];
						if (!tc || tc.type !== "toolCall") continue;
						if (tc.name === "write") {
							const targetPath = resolvedLoopPathForTool(tc);
							if (!targetPath) continue;
							if (tr.isError) {
								if (pendingMessages.length === 0) pendingMessages.push({ role: "user", content: [{ type: "text", text: `Write failed for \`${targetPath}\`. Check path and arguments; retry with \`write\` or switch to \`edit\` on an existing file.` }], timestamp: Date.now() });
								continue;
							}
							await recordSuccessfulFileMutation(targetPath);
							continue;
						}
						if (tc.name !== "edit") continue;
						const targetPath = resolvedLoopPathForTool(tc);
						if (!targetPath) continue;
						if (tr.isError) {
							const errText = tr.content?.map((c: any) => c.text ?? "").join("") ?? "";
							if (pendingMessages.length === 0) {
								pendingMessages.push({
									role: "user",
									content: [{ type: "text", text: buildGeminiEditFailureRecoveryMessage(targetPath, errText, tc, foundFiles) }],
									timestamp: Date.now(),
								});
							}
						} else {
							await recordSuccessfulFileMutation(targetPath);
							refreshNetChangedStateForPath(targetPath);
						}
					}

					for (let i = 0; i < toolResults.length; i++) {
						const tr = toolResults[i];
						const tc = toolCalls[i];
						if (tr.toolName !== "read" || tr.isError || !tc || tc.type !== "toolCall") continue;
						const readPath = resolvedLoopPathForTool(tc);
						if (!readPath) continue;
						const rArgs = tc.arguments as { offset?: number; limit?: number } | undefined;
						const isFullRead = typeof rArgs?.offset === "undefined" && typeof rArgs?.limit === "undefined";
						if (!isFullRead) continue;
						addReadPathVariants(pathsAlreadyRead, readPath);
						const normRead = normalizePathForMatch(readPath);
						const readText = tr.content?.map((c: any) => c?.text ?? "").join("") ?? "";
						if (!baselineFileContentByPath.has(normRead) && readText.length > 0) {
							baselineFileContentByPath.set(normRead, readText);
						}
						const planText = planByFile.get(normRead) ?? "";
						if (pendingMessages.length === 0) {
							pendingMessages.push({
								role: "user",
								content: [{ type: "text", text: `IMPLEMENT read loaded \`${readPath}\`. Next action must be \`edit\`/\`write\` for planned changes only.${planText ? `\n\nPlanned edits for this file:\n${planText}` : ""}\n\nMatch existing local style (naming, formatting, surrounding patterns) in this file.` }],
								timestamp: Date.now(),
							});
						}
					}
					if (confirmPhaseStarted && confirmCurrentPath && hasMoreToolCalls && pendingMessages.length === 0) {
						pendingMessages.push(buildConfirmPrompt(confirmCurrentPath));
					}
				}
			}
			const noProgressSinceSignature = successfulMutationCount === lastSignatureMutationCount;
			if (
				hasProducedEdit &&
				repeatedToolSignatureCount >= REPEATED_TOOL_SIGNATURE_LIMIT &&
				noProgressSinceSignature &&
				planSubmitted &&
				pendingMessages.length === 0
			) {
				if (executionMode === "implement" && netChangedPaths.size === 0) {
					pendingMessages.push({
						role: "user",
						content: [{
							type: "text",
							text: "No net persisted edits detected yet. Continue implement mode with concrete edit/write calls that leave actual file changes.",
						}],
						timestamp: Date.now(),
					});
					continue;
				}
				await emit({ type: "turn_end", message, toolResults });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			await emit({ type: "turn_end", message, toolResults });

			// Preserve in-loop nudges (e.g. PLAN->IMPLEMENT handoff) and append external steering.
			// Overwriting here can drop critical transition instructions and cause no-op implement turns.
			const steeringMessages = (await config.getSteeringMessages?.()) || [];
			if (steeringMessages.length > 0) {
				pendingMessages.push(...steeringMessages);
			}

			// Do not cut off mid-session right after tools when exploration already burned the budget (task11-style:
			// long reads then one edit). Only apply the soft time limit when the model ends a turn with no tool calls.
			if (
				!hasMoreToolCalls &&
				pendingMessages.length === 0 &&
				executionMode === "implement" &&
				hasProducedEdit &&
				missingPlannedFiles().length === 0 &&
				implementModeStartedAt !== null &&
				(Date.now() - implementModeStartedAt) >= GRACEFUL_EXIT_MS
			) {
				if (netChangedPaths.size === 0) {
					pendingMessages.push({
						role: "user",
						content: [{
							type: "text",
							text: "No net file changes are currently detected versus earlier reads. Do not stop. Re-read target planned files and apply concrete edits that persist on disk.",
						}],
						timestamp: Date.now(),
					});
					continue;
				}
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			pendingMessages = followUpMessages;
			continue;
		}
		// Hard cutoff for implementation flow: stop after 200s.
		if (
			executionMode === "implement" &&
			implementModeStartedAt !== null &&
			(Date.now() - implementModeStartedAt) >= IMPLEMENT_VERIFY_MAX_MS
		) {
			break;
		}
		// IMPORTANT: In IMPLEMENT mode, plans are the frozen contract. Do NOT nudge the model
		// about task-derived "expected/explicit/criterion" files, because they can include
		// files that were never planned (e.g., sibling modules surfaced by PLAN-mode discovery).
		// Any off-plan nudge here distracts the model and forces out-of-scope edits.
		// Plan-scoped exit gates below (missingPlannedFiles / needsDeeperPlannedImplementation)
		// are the ONLY allowed post-edit continue signals in IMPLEMENT mode.
		if (executionMode === "plan") {
			pendingMessages = [{
				role: "user",
				content: [{
					type: "text",
					text: "Cannot stop in PLAN mode. Continue exploration and submit final file-by-file plan via `plan` tool.",
				}],
				timestamp: Date.now(),
			}];
			continue;
		} else if (executionMode === "implement") {
			if (!planSubmitted) {
				await emitRolloutMarker("mode_transition", { from: "implement", to: "plan", reason: "invariant_correction" });
				executionMode = "plan";
				confirmPhaseStarted = false;
				confirmPhaseDone = false;
				confirmCurrentPath = null;
				confirmPlanList.clear();
				pendingMessages = [{
					role: "user",
					content: [{
						type: "text",
						text: "Invariant correction: implement mode requires successful `plan` tool call first. Return to PLAN mode, finish planning, and call `plan` tool with exact JSON format.",
					}],
					timestamp: Date.now(),
				}];
				continue;
			}
			const missingFromPlan = missingPlannedFiles();
			if (missingFromPlan.length > 0) {
				pendingMessages = [{
					role: "user",
					content: [{
						type: "text",
						text: `Do not stop. Planned files still unedited: ${missingFromPlan.slice(0, 10).map((f) => `\`${f}\``).join(", ")}. Implement all planned files.`,
					}],
					timestamp: Date.now(),
				}];
				continue;
			}
			const shallowPlanned = needsDeeperPlannedImplementation();
			if (shallowPlanned.length > 0) {
				pendingMessages = [{
					role: "user",
					content: [{
						type: "text",
						text: `Do not stop yet. These planned files have high-detail plans but too little implementation evidence: ${shallowPlanned
							.slice(0, 8)
							.map((f) => `\`${f}\``)
							.join(", ")}. Continue implementing those plan details before stopping.`,
					}],
					timestamp: Date.now(),
				}];
				continue;
			}
			if (!confirmPhaseDone) {
				if (!confirmPhaseStarted) {
					confirmPhaseStarted = true;
					for (const pf of plannedFiles) {
						if (isPathEdited(pf)) confirmPlanList.add(normalizePathForMatch(pf));
					}
				}
				if (confirmCurrentPath === null && queueNextConfirmPrompt()) {
					continue;
				}
			}
		}
		// Review pass: if finished quickly and edits were made, check for missed files
		const reviewElapsed = Date.now() - loopStart;
		// Previously capped at 60s, which skipped the review nudge on slow models (exploration alone could exceed it).
		if (!reviewPassDone && hasProducedEdit && reviewElapsed < ABSOLUTE_SESSION_CAP_MS) {
			reviewPassDone = true;
			const uneditedTargets = foundFiles.filter(
				(f: string) => {
					const nf = f.replace(/^\.\//, "");
					return !editedPaths.has(f) && !editedPaths.has(nf) && !editedPaths.has("./" + nf);
				}
			);
			const hint = uneditedTargets.length > 0
				? `Unedited discovered files: ${uneditedTargets.slice(0, 5).map((f: string) => `\`${f}\``).join(", ")}. Check whether any still map to unmet acceptance criteria; edit only the required ones.`
				: `Re-read the task acceptance criteria. If the task listed exact old strings or labels, grep the repo for any that remain. Are there files or criteria you missed? If yes, discover and edit them. If all criteria are covered, reply "done".`;
			pendingMessages = [{
				role: "user",
				content: [{ type: "text", text: `REVIEW: You edited ${editedPaths.size} file(s): ${[...editedPaths].slice(0, 8).join(", ")}. ${hint}` }],
				timestamp: Date.now(),
			}];
			continue;
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
): Promise<ToolResultMessage[]> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	if (config.toolExecution === "sequential") {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
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
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit));
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
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit));
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

	return await emitToolCallOutcome(prepared.toolCall, result, isError, emit);
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
		timestamp: Date.now(),
	};

	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
	return toolResultMessage;
}
