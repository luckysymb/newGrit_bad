import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Container, Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "@sinclair/typebox";
import { constants } from "fs";
import { access as fsAccess, readFile as fsReadFile, writeFile as fsWriteFile } from "fs/promises";
import { renderDiff } from "../../modes/interactive/components/diff.js";
import type { ToolDefinition } from "../extensions/types.js";
import {
	applyLineRangeEditsToNormalizedContent,
	detectLineEnding,
	generateDiffString,
	type LineRangeEdit,
	normalizeToLF,
	restoreLineEndings,
	stripBom,
} from "./edit-diff.js";
import { withFileMutationQueue } from "./file-mutation-queue.js";
import { resolveToCwd, resolveWorkspacePath } from "./path-utils.js";
import { invalidArgText, shortenPath, str } from "./render-utils.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";

type EditRenderState = Record<string, never>;

/**
 * Line-range based replacement. The model specifies WHICH lines to replace
 * (by 0-indexed inclusive startLine/endLine) and WHAT to replace them with.
 * `oldText` is used as a coarse verification guard: the tool reads the real
 * lines at [startLine..endLine] and compares the alphanumeric projection with
 * `oldText` before doing the replacement.
 *
 * The schema is intentionally FLAT (no nested tuple array). Flat primitive
 * fields are what providers (Gemini, OpenAI, Anthropic) emit most reliably
 * inside an array of objects; nested tuple schemas cause some providers to
 * return `edits: []` and give up entirely.
 */
const replaceEditSchema = Type.Object(
	{
		startLine: Type.Integer({
			minimum: 0,
			description:
				"Zero-indexed INCLUSIVE first line of the range to replace. Line 0 is the first line of the file. Refers to the ORIGINAL file content (before any edit in this call is applied).",
		}),
		endLine: Type.Integer({
			minimum: 0,
			description:
				"Zero-indexed INCLUSIVE last line of the range to replace. For a single-line edit set endLine equal to startLine. Refers to the ORIGINAL file content.",
		}),
		oldText: Type.String({
			description:
				"The ACTUAL current text of lines [startLine..endLine] INCLUSIVE, joined by \\n, COPIED VERBATIM from a fresh `read` of the file. Used as a verification guard — the tool compares ONLY alphanumeric characters ([a-zA-Z0-9]) between oldText and the real lines, so whitespace, indentation, punctuation, quotes, and trailing newlines do NOT need to match exactly. Do NOT send an empty string, a single space, or a placeholder — always copy the real content of every line in the range. To insert new content without removing anything, include the original target line in both `oldText` and `newText` (prepending/appending your addition inside `newText`).",
		}),
		newText: Type.String({
			description:
				"Replacement text for lines [startLine..endLine]. May contain multiple lines (use \\n). Pass an empty string to delete the range.",
		}),
	},
	{ additionalProperties: false },
);

const editSchema = Type.Object(
	{
		path: Type.String({ description: "Path to the file to edit (relative or absolute)" }),
		edits: Type.Array(replaceEditSchema, {
			description:
				"One or more line-range replacements. Each entry has {startLine, endLine, oldText, newText} as flat fields (no nested array). All startLine/endLine values refer to the ORIGINAL file. Ranges must be non-overlapping. The tool applies them in reverse order so earlier line numbers stay valid.",
		}),
	},
	{ additionalProperties: false },
);

export type EditToolInput = Static<typeof editSchema>;

export interface EditToolDetails {
	/** Unified diff of the changes made */
	diff: string;
	/** Line number of the first change in the new file (for editor navigation) */
	firstChangedLine?: number;
}

/**
 * Pluggable operations for the edit tool.
 * Override these to delegate file editing to remote systems (for example SSH).
 */
export interface EditOperations {
	/** Read file contents as a Buffer */
	readFile: (absolutePath: string) => Promise<Buffer>;
	/** Write content to a file */
	writeFile: (absolutePath: string, content: string) => Promise<void>;
	/** Check if file is readable and writable (throw if not) */
	access: (absolutePath: string) => Promise<void>;
}

const defaultEditOperations: EditOperations = {
	readFile: (path) => fsReadFile(path),
	writeFile: (path, content) => fsWriteFile(path, content, "utf-8"),
	access: (path) => fsAccess(path, constants.R_OK | constants.W_OK),
};

export interface EditToolOptions {
	/** Custom operations for file editing. Default: local filesystem */
	operations?: EditOperations;
}

/**
 * Prepared entries use the SAME shape as the public schema
 * ({ startLine, endLine, oldText, newText }) so that the wrapper's JSON-schema
 * validator accepts them. Internal conversion to LineRangeEdit happens later
 * inside `validateEditInput`, which runs after schema validation.
 */
type PreparedEditEntry = {
	startLine: number;
	endLine: number;
	oldText: string;
	newText: string;
};

function isFiniteInt(n: unknown): n is number {
	return typeof n === "number" && Number.isInteger(n) && Number.isFinite(n);
}

function coerceLineRange(value: unknown): [number, number] | null {
	if (!Array.isArray(value) || value.length !== 2) return null;
	const first = value[0];
	const last = value[1];
	if (!isFiniteInt(first) || !isFiniteInt(last)) return null;
	if (first < 0 || last < first) return null;
	return [first, last];
}

/**
 * Extract [startLine, endLine] from a raw edit entry, accepting several aliases
 * so that providers with different schema-filling styles still succeed:
 *   - Preferred flat form: { startLine, endLine }
 *   - Tuple form:          { lineRange: [start, end] }
 *   - Snake_case:          { start_line, end_line }
 *   - firstLine/lastLine:  { firstLine, lastLine } / { first_line, last_line }
 */
function extractLineRange(r: Record<string, unknown>): [number, number] | null {
	const pairs: Array<[unknown, unknown]> = [
		[r.startLine, r.endLine],
		[r.start_line, r.end_line],
		[r.firstLine, r.lastLine],
		[r.first_line, r.last_line],
	];
	for (const [a, b] of pairs) {
		if (isFiniteInt(a) && isFiniteInt(b) && a >= 0 && b >= a) {
			return [a, b];
		}
	}
	const tuple = coerceLineRange(r.lineRange);
	if (tuple) return tuple;
	return null;
}

function coerceEditEntry(raw: unknown): PreparedEditEntry | null {
	if (!raw || typeof raw !== "object") return null;
	const r = raw as Record<string, unknown>;
	const range = extractLineRange(r);
	if (!range) return null;
	if (typeof r.oldText !== "string") return null;
	if (typeof r.newText !== "string") return null;
	return { startLine: range[0], endLine: range[1], oldText: r.oldText, newText: r.newText };
}

function pickEditPath(args: { path?: unknown; file_path?: unknown }): string | undefined {
	if (typeof args.path === "string") {
		const t = args.path.trim();
		if (t.length > 0) return t;
	}
	if (typeof args.file_path === "string") {
		const t = args.file_path.trim();
		if (t.length > 0) return t;
	}
	return undefined;
}

/**
 * Drop malformed or partial `edits[]` entries (common when JSON is truncated) so a single
 * valid replacement can still run instead of failing schema validation on the whole call.
 */
function prepareEditArguments(input: unknown): EditToolInput {
	if (!input || typeof input !== "object") {
		return input as EditToolInput;
	}

	const args = input as {
		path?: unknown;
		file_path?: unknown;
		edits?: unknown[];
		startLine?: unknown;
		endLine?: unknown;
		lineRange?: unknown;
		oldText?: unknown;
		newText?: unknown;
	};
	const collected: PreparedEditEntry[] = [];

	if (Array.isArray(args.edits)) {
		for (const entry of args.edits) {
			const e = coerceEditEntry(entry);
			if (e) collected.push(e);
		}
	}

	// Tolerate a flat top-level shape too: { startLine, endLine, oldText, newText }
	// (or any of the aliases supported by extractLineRange).
	if (typeof args.oldText === "string" && typeof args.newText === "string") {
		const topRange = extractLineRange(args as Record<string, unknown>);
		if (topRange) {
			collected.push({
				startLine: topRange[0],
				endLine: topRange[1],
				oldText: args.oldText,
				newText: args.newText,
			});
		}
	}

	if (collected.length === 0) {
		return input as EditToolInput;
	}

	const path = pickEditPath(args);
	if (path === undefined) {
		return input as EditToolInput;
	}

	return { path, edits: collected } as unknown as EditToolInput;
}

function validateEditInput(input: EditToolInput): { path: string; edits: LineRangeEdit[] } {
	if (!Array.isArray(input.edits) || input.edits.length === 0) {
		throw new Error("Edit tool input is invalid. edits must contain at least one replacement.");
	}
	const normalized: LineRangeEdit[] = [];
	for (let i = 0; i < input.edits.length; i++) {
		const entry = input.edits[i] as unknown as Record<string, unknown>;
		const range = extractLineRange(entry ?? {}) ?? [0, 0];
		const oldText = typeof entry?.oldText === "string" ? (entry.oldText as string) : "";
		const newText = typeof entry?.newText === "string" ? (entry.newText as string) : "";
		normalized.push({ lineRange: range, oldText, newText });
	}
	return { path: input.path, edits: normalized };
}

type RenderableEditArgs = {
	path?: string;
	file_path?: string;
	edits?: Array<{
		startLine?: number;
		endLine?: number;
		lineRange?: number[];
		oldText?: string;
		newText?: string;
	}>;
};

function formatEditCall(
	args: RenderableEditArgs | undefined,
	theme: typeof import("../../modes/interactive/theme/theme.js").theme,
): string {
	const invalidArg = invalidArgText(theme);
	const rawPath = str(args?.file_path ?? args?.path);
	const path = rawPath !== null ? shortenPath(rawPath) : null;
	const pathDisplay = path === null ? invalidArg : path ? theme.fg("accent", path) : theme.fg("toolOutput", "...");
	return `${theme.fg("toolTitle", theme.bold("edit"))} ${pathDisplay}`;
}

function formatEditResult(
	args: RenderableEditArgs | undefined,
	result: {
		content: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
		details?: EditToolDetails;
	},
	theme: typeof import("../../modes/interactive/theme/theme.js").theme,
	isError: boolean,
): string | undefined {
	const rawPath = str(args?.file_path ?? args?.path);
	if (isError) {
		const errorText = result.content
			.filter((c) => c.type === "text")
			.map((c) => c.text || "")
			.join("\n");
		if (!errorText) {
			return undefined;
		}
		return `\n${theme.fg("error", errorText)}`;
	}

	const resultDiff = result.details?.diff;
	if (!resultDiff) {
		return undefined;
	}
	return `\n${renderDiff(resultDiff, { filePath: rawPath ?? undefined })}`;
}

export function createEditToolDefinition(
	cwd: string,
	options?: EditToolOptions,
): ToolDefinition<typeof editSchema, EditToolDetails | undefined, EditRenderState> {
	const ops = options?.operations ?? defaultEditOperations;
	return {
		name: "edit",
		label: "edit",
		description:
			"Edit a file by REPLACING a zero-indexed inclusive line range [startLine..endLine] with newText. Each entry in `edits` is a flat object {startLine, endLine, oldText, newText}. The tool TRUSTS the line numbers — out-of-range values are silently clamped (use startLine >= file length to append). oldText is the only guard: always copy the real content of every line in the range.",
		promptSnippet:
			"Replace file lines by zero-indexed inclusive {startLine,endLine}; oldText is a flexible lowercase-alnum sanity guard.",
		promptGuidelines: [
			"Line numbers are 0-indexed and endLine is INCLUSIVE. Single line: endLine = startLine.",
			"oldText is the only guard. Do NOT send an empty string, a single space, or a placeholder — always copy the real content of every line in the range.",
			"Pass newText=\"\" to delete a range. To append at end of file, use startLine = (file length).",
		],
		parameters: editSchema,
		prepareArguments: prepareEditArguments,
		async execute(_toolCallId, input: EditToolInput, signal?: AbortSignal, _onUpdate?, _ctx?) {
			const { path, edits } = validateEditInput(input);
			const resolvedPath = resolveWorkspacePath(path, cwd, { kind: "file", basenameFallback: true });
			const absolutePath = resolveToCwd(resolvedPath, cwd);

			return withFileMutationQueue(
				absolutePath,
				() =>
					new Promise<{
						content: Array<{ type: "text"; text: string }>;
						details: EditToolDetails | undefined;
					}>((resolve, reject) => {
						if (signal?.aborted) {
							reject(new Error("Operation aborted"));
							return;
						}

						let aborted = false;

						const onAbort = () => {
							aborted = true;
							reject(new Error("Operation aborted"));
						};

						if (signal) {
							signal.addEventListener("abort", onAbort, { once: true });
						}

						void (async () => {
							try {
								try {
									await ops.access(absolutePath);
								} catch {
									if (signal) {
										signal.removeEventListener("abort", onAbort);
									}
									reject(new Error(`File not found: ${resolvedPath}`));
									return;
								}

								if (aborted) {
									return;
								}

								const buffer = await ops.readFile(absolutePath);
								const rawContent = buffer.toString("utf-8");

								if (aborted) {
									return;
								}

								const { bom, text: content } = stripBom(rawContent);
								const originalEnding = detectLineEnding(content);
								const normalizedContent = normalizeToLF(content);

								const { baseContent, newContent } = applyLineRangeEditsToNormalizedContent(
									normalizedContent,
									edits,
									resolvedPath,
								);

								if (aborted) {
									return;
								}

								const changed = baseContent !== newContent;
								if (changed) {
									const finalContent = bom + restoreLineEndings(newContent, originalEnding);
									await ops.writeFile(absolutePath, finalContent);
								}

								if (aborted) {
									return;
								}

								if (signal) {
									signal.removeEventListener("abort", onAbort);
								}

								const diffResult = generateDiffString(baseContent, newContent);
								const rangeSummary = edits
									.map((e) => `[${e.lineRange[0]}-${e.lineRange[1]}]`)
									.join(", ");
								const summary = `Successfully replaced ${edits.length} line range(s) ${rangeSummary} in ${resolvedPath}.`;
								resolve({
									content: [
										{
											type: "text",
											text: summary,
										},
									],
									details: { diff: diffResult.diff, firstChangedLine: diffResult.firstChangedLine },
								});
							} catch (error: unknown) {
								if (signal) {
									signal.removeEventListener("abort", onAbort);
								}

								if (!aborted) {
									reject(error instanceof Error ? error : new Error(String(error)));
								}
							}
						})();
					}),
			);
		},
		renderCall(args, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			text.setText(formatEditCall(args, theme));
			return text;
		},
		renderResult(result, _options, theme, context) {
			const output = formatEditResult(context.args, result as any, theme, context.isError);
			if (!output) {
				const component = (context.lastComponent as Container | undefined) ?? new Container();
				component.clear();
				return component;
			}
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			text.setText(output);
			return text;
		},
	};
}

export function createEditTool(cwd: string, options?: EditToolOptions): AgentTool<typeof editSchema> {
	return wrapToolDefinition(createEditToolDefinition(cwd, options));
}

/** Default edit tool using process.cwd() for backwards compatibility. */
export const editToolDefinition = createEditToolDefinition(process.cwd());
export const editTool = createEditTool(process.cwd());
