import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Container, Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "@sinclair/typebox";
import { constants } from "fs";
import { access as fsAccess, readFile as fsReadFile, writeFile as fsWriteFile } from "fs/promises";
import { renderDiff } from "../../modes/interactive/components/diff.js";
import type { ToolDefinition } from "../extensions/types.js";
import {
	applyEditsToNormalizedContent,
	detectLineEnding,
	type Edit,
	fuzzyFindText,
	generateDiffString,
	normalizeToLF,
	restoreLineEndings,
	stripBom,
} from "./edit-diff.js";
import { withFileMutationQueue } from "./file-mutation-queue.js";
import { resolveToCwd, resolveWorkspacePath } from "./path-utils.js";
import { invalidArgText, shortenPath, str } from "./render-utils.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";

type EditRenderState = Record<string, never>;

const replaceEditSchema = Type.Object(
	{
		oldText: Type.String({
			description:
				"Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.",
		}),
		newText: Type.String({ description: "Replacement text for this targeted edit." }),
	},
	{ additionalProperties: false },
);

const editSchema = Type.Object(
	{
		path: Type.String({ description: "Path to the file to edit (relative or absolute)" }),
		edits: Type.Array(replaceEditSchema, {
			description:
				"One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
		}),
	},
	{ additionalProperties: false },
);

export type EditToolInput = Static<typeof editSchema>;
type LegacyEditToolInput = EditToolInput & {
	oldText?: unknown;
	newText?: unknown;
};

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

function isValidEditPair(o: unknown): o is { oldText: string; newText: string } {
	if (!o || typeof o !== "object") return false;
	const e = o as Record<string, unknown>;
	if (typeof e.oldText !== "string" || typeof e.newText !== "string") return false;
	// Empty oldText always fails apply; drop so other valid edits in the same call can run.
	if (e.oldText.length === 0) return false;
	return true;
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

	const args = input as LegacyEditToolInput & { file_path?: unknown; edits?: unknown[] };
	const collected: Array<{ oldText: string; newText: string }> = [];

	if (Array.isArray(args.edits)) {
		for (const entry of args.edits) {
			if (isValidEditPair(entry)) {
				collected.push({ oldText: entry.oldText, newText: entry.newText });
			}
		}
	}

	if (typeof args.oldText === "string" && typeof args.newText === "string" && args.oldText.length > 0) {
		collected.push({ oldText: args.oldText, newText: args.newText });
	}

	if (collected.length === 0) {
		return input as EditToolInput;
	}

	const path = pickEditPath(args);
	if (path === undefined) {
		return input as EditToolInput;
	}

	return { path, edits: collected } as EditToolInput;
}

function validateEditInput(input: EditToolInput): { path: string; edits: Edit[] } {
	if (!Array.isArray(input.edits) || input.edits.length === 0) {
		throw new Error("Edit tool input is invalid. edits must contain at least one replacement.");
	}
	return { path: input.path, edits: input.edits };
}

function extractFailedEditIndex(message: string): number | null {
	const m = message.match(/edits\[(\d+)\]/);
	if (!m) return null;
	const n = Number.parseInt(m[1], 10);
	return Number.isFinite(n) ? n : null;
}

/**
 * Multi-edit calls are all-or-nothing by default. When one anchor drifts (common with large model edits),
 * salvage the call by keeping only edits that individually match against the original content.
 */
function tryApplyEditsWithSalvage(
	normalizedContent: string,
	edits: Edit[],
	path: string,
): {
	baseContent: string;
	newContent: string;
	appliedCount: number;
	skippedCount: number;
	alreadyAppliedCount: number;
} {
	try {
		const { baseContent, newContent } = applyEditsToNormalizedContent(normalizedContent, edits, path);
		return { baseContent, newContent, appliedCount: edits.length, skippedCount: 0, alreadyAppliedCount: 0 };
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		const failedIndex = extractFailedEditIndex(message);
		if (edits.length <= 1 || failedIndex === null || !message.includes("Could not find")) {
			throw error;
		}

		const validEdits: Edit[] = [];
		let alreadyAppliedCount = 0;
		for (let i = 0; i < edits.length; i++) {
			const candidate = edits[i];
			try {
				// Probe each edit in isolation against the original content.
				applyEditsToNormalizedContent(normalizedContent, [candidate], path);
				validEdits.push(candidate);
			} catch {
				const oldMatch = fuzzyFindText(normalizedContent, normalizeToLF(candidate.oldText)).found;
				const newMatch = fuzzyFindText(normalizedContent, normalizeToLF(candidate.newText)).found;
				const oldEqNew = normalizeToLF(candidate.oldText) === normalizeToLF(candidate.newText);
				if (!oldEqNew && !oldMatch && newMatch) {
					alreadyAppliedCount++;
				}
			}
		}
		if (validEdits.length === 0 && alreadyAppliedCount === 0) {
			throw error;
		}
		if (validEdits.length === 0) {
			// Fully idempotent replay: requested changes are already present on disk.
			return {
				baseContent: normalizedContent,
				newContent: normalizedContent,
				appliedCount: 0,
				skippedCount: edits.length - alreadyAppliedCount,
				alreadyAppliedCount,
			};
		}
		const { baseContent, newContent } = applyEditsToNormalizedContent(normalizedContent, validEdits, path);
		return {
			baseContent,
			newContent,
			appliedCount: validEdits.length,
			skippedCount: edits.length - validEdits.length,
			alreadyAppliedCount,
		};
	}
}

type RenderableEditArgs = {
	path?: string;
	file_path?: string;
	edits?: Edit[];
	oldText?: string;
	newText?: string;
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
			"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
		promptSnippet:
			"Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
		promptGuidelines: [
			"Use edit for precise changes (edits[].oldText must match exactly)",
			"When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
			"Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
			"Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
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
						// Check if already aborted.
						if (signal?.aborted) {
							reject(new Error("Operation aborted"));
							return;
						}

						let aborted = false;

						// Set up abort handler.
						const onAbort = () => {
							aborted = true;
							reject(new Error("Operation aborted"));
						};

						if (signal) {
							signal.addEventListener("abort", onAbort, { once: true });
						}

						// Perform the edit operation.
						void (async () => {
							try {
								// Check if file exists.
								try {
									await ops.access(absolutePath);
								} catch {
									if (signal) {
										signal.removeEventListener("abort", onAbort);
									}
									reject(new Error(`File not found: ${resolvedPath}`));
									return;
								}

								// Check if aborted before reading.
								if (aborted) {
									return;
								}

								// Read the file.
								const buffer = await ops.readFile(absolutePath);
								const rawContent = buffer.toString("utf-8");

								// Check if aborted after reading.
								if (aborted) {
									return;
								}

								// Strip BOM before matching. The model will not include an invisible BOM in oldText.
								const { bom, text: content } = stripBom(rawContent);
								const originalEnding = detectLineEnding(content);
								const normalizedContent = normalizeToLF(content);
								const { baseContent, newContent, appliedCount, skippedCount, alreadyAppliedCount } =
									tryApplyEditsWithSalvage(
									normalizedContent,
									edits,
									resolvedPath,
								);

								// Check if aborted before writing.
								if (aborted) {
									return;
								}

								const changed = baseContent !== newContent;
								if (changed) {
									const finalContent = bom + restoreLineEndings(newContent, originalEnding);
									await ops.writeFile(absolutePath, finalContent);
								}

								// Check if aborted after writing.
								if (aborted) {
									return;
								}

								// Clean up abort handler.
								if (signal) {
									signal.removeEventListener("abort", onAbort);
								}

								const diffResult = generateDiffString(baseContent, newContent);
								const summary =
									!changed && alreadyAppliedCount > 0
										? `No-op: ${alreadyAppliedCount}/${edits.length} block(s) already applied in ${resolvedPath}.`
										: skippedCount > 0
											? `Partially applied ${appliedCount}/${edits.length} block(s) in ${resolvedPath}. ${skippedCount} block(s) were skipped due to stale/non-matching oldText.`
											: `Successfully replaced ${appliedCount} block(s) in ${resolvedPath}.`;
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
								// Clean up abort handler.
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
