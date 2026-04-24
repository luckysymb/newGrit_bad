/**
 * Shared diff computation utilities for the edit tool.
 * Used by both edit.ts (for execution) and tool-execution.ts (for preview rendering).
 */

import * as Diff from "diff";
import { constants } from "fs";
import { access, readFile } from "fs/promises";
import { resolveToCwd, resolveWorkspacePath } from "./path-utils.js";

export function detectLineEnding(content: string): "\r\n" | "\n" {
	const crlfIdx = content.indexOf("\r\n");
	const lfIdx = content.indexOf("\n");
	if (lfIdx === -1) return "\n";
	if (crlfIdx === -1) return "\n";
	return crlfIdx < lfIdx ? "\r\n" : "\n";
}

export function normalizeToLF(text: string): string {
	return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

export function restoreLineEndings(text: string, ending: "\r\n" | "\n"): string {
	return ending === "\r\n" ? text.replace(/\n/g, "\r\n") : text;
}

/**
 * Normalize text for fuzzy matching. Applies progressive transformations:
 * - Strip trailing whitespace from each line
 * - Normalize smart quotes to ASCII equivalents
 * - Normalize Unicode dashes/hyphens to ASCII hyphen
 * - Normalize special Unicode spaces to regular space
 */
export function normalizeForFuzzyMatch(text: string): string {
	return (
		text
			.normalize("NFKC")
			// Models often mix tabs with spaces; normalize tabs for comparison only.
			.replace(/\t/g, "    ")
			// Zero-width / invisible characters that break naive string equality
			.replace(/[\u200B\u200C\u200D\uFEFF]/g, "")
			// Strip trailing whitespace per line
			.split("\n")
			.map((line) => line.trimEnd())
			.join("\n")
			// Smart single quotes → '
			.replace(/[\u2018\u2019\u201A\u201B]/g, "'")
			// Smart double quotes → "
			.replace(/[\u201C\u201D\u201E\u201F]/g, '"')
			// Various dashes/hyphens → -
			// U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash,
			// U+2013 en-dash, U+2014 em-dash, U+2015 horizontal bar, U+2212 minus
			.replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]/g, "-")
			// Special spaces → regular space
			// U+00A0 NBSP, U+2002-U+200A various spaces, U+202F narrow NBSP,
			// U+205F medium math space, U+3000 ideographic space
			.replace(/[\u00A0\u2002-\u200A\u202F\u205F\u3000]/g, " ")
	);
}

/**
 * Find non-overlapping regions where each line matches needle lines after .trim() on both sides.
 * Used when literal and normalized substring search fail (indent/whitespace drift).
 */
function findTrimmedLineBlocks(haystack: string, needle: string): Array<{ start: number; end: number }> {
	const hLines = haystack.split("\n");
	const nLines = needle.split("\n");
	if (nLines.length === 0 || (nLines.length === 1 && nLines[0] === "")) {
		return [];
	}
	const results: Array<{ start: number; end: number }> = [];
	for (let i = 0; i <= hLines.length - nLines.length; i++) {
		let ok = true;
		for (let j = 0; j < nLines.length; j++) {
			if (hLines[i + j].trim() !== nLines[j].trim()) {
				ok = false;
				break;
			}
		}
		if (!ok) continue;
		let start = 0;
		for (let k = 0; k < i; k++) {
			start += hLines[k].length + 1;
		}
		let end = start;
		for (let k = 0; k < nLines.length; k++) {
			end += hLines[i + k].length;
			if (k < nLines.length - 1) end += 1;
		}
		results.push({ start, end });
	}
	return results;
}

/** Collapse runs of whitespace (incl. NBSP) for line comparison — helps HTML/markup where spacing drifts. */
function collapseLineWhitespace(line: string): string {
	return line.replace(/\s+/g, " ").trim();
}

/**
 * Like findTrimmedLineBlocks, but lines match if their whitespace-collapsed forms are equal.
 */
function findRelaxedWhitespaceLineBlocks(haystack: string, needle: string): Array<{ start: number; end: number }> {
	const hLines = haystack.split("\n");
	const nLines = needle.split("\n");
	if (nLines.length === 0 || (nLines.length === 1 && nLines[0] === "")) {
		return [];
	}
	const results: Array<{ start: number; end: number }> = [];
	for (let i = 0; i <= hLines.length - nLines.length; i++) {
		let ok = true;
		for (let j = 0; j < nLines.length; j++) {
			if (collapseLineWhitespace(hLines[i + j]) !== collapseLineWhitespace(nLines[j])) {
				ok = false;
				break;
			}
		}
		if (!ok) continue;
		let start = 0;
		for (let k = 0; k < i; k++) {
			start += hLines[k].length + 1;
		}
		let end = start;
		for (let k = 0; k < nLines.length; k++) {
			end += hLines[i + k].length;
			if (k < nLines.length - 1) end += 1;
		}
		results.push({ start, end });
	}
	return results;
}

/**
 * Conservative fallback for large near-miss blocks:
 * If oldText is long, match a unique range from first and last non-empty lines.
 * This avoids hard failures when the model paraphrases a few interior lines.
 */
function findAnchorRangeBlocks(haystack: string, needle: string): Array<{ start: number; end: number }> {
	const hLines = haystack.split("\n");
	const nLines = needle.split("\n");
	if (nLines.length < 6) return [];
	const firstIdx = nLines.findIndex((l) => l.trim().length > 0);
	const lastIdx = (() => {
		for (let i = nLines.length - 1; i >= 0; i--) {
			if (nLines[i].trim().length > 0) return i;
		}
		return -1;
	})();
	if (firstIdx === -1 || lastIdx === -1 || firstIdx >= lastIdx) return [];
	const firstLine = collapseLineWhitespace(nLines[firstIdx]);
	const lastLine = collapseLineWhitespace(nLines[lastIdx]);
	if (!firstLine || !lastLine) return [];
	const out: Array<{ start: number; end: number }> = [];
	for (let i = 0; i < hLines.length; i++) {
		if (collapseLineWhitespace(hLines[i]) !== firstLine) continue;
		for (let j = i + 1; j < hLines.length; j++) {
			if (collapseLineWhitespace(hLines[j]) !== lastLine) continue;
			const span = j - i + 1;
			// Keep fallback conservative: span should be reasonably close to target block size.
			if (span < Math.max(3, nLines.length - 40) || span > nLines.length + 120) continue;
			let start = 0;
			for (let k = 0; k < i; k++) start += hLines[k].length + 1;
			let end = start;
			for (let k = i; k <= j; k++) {
				end += hLines[k].length;
				if (k < j) end += 1;
			}
			out.push({ start, end });
		}
	}
	return out;
}

/** Try a few newline variants models often get wrong vs on-disk files */
function newlineVariants(s: string): string[] {
	const out: string[] = [s];
	if (s.endsWith("\n")) {
		out.push(s.slice(0, -1));
	} else {
		out.push(`${s}\n`);
	}
	return [...new Set(out)];
}

export interface FuzzyMatchResult {
	/** Whether a match was found */
	found: boolean;
	/** The index where the match starts (in the content that should be used for replacement) */
	index: number;
	/** Length of the matched text */
	matchLength: number;
	/** Whether fuzzy matching was used (false = exact match) */
	usedFuzzyMatch: boolean;
	/**
	 * The content to use for replacement operations.
	 * When exact match: original content. When fuzzy match: normalized content.
	 */
	contentForReplacement: string;
}

export interface Edit {
	oldText: string;
	newText: string;
}

/**
 * Line-range based edit (0-indexed, inclusive [firstLine, lastLine]).
 *
 * The tool replaces the exact lines at [firstLine..lastLine] with `newText`, using
 * `oldText` as a verification guard to confirm the model is editing the lines it
 * thinks it is. `oldText` is compared against the joined content of the targeted
 * lines with line-ending normalization and trailing-whitespace tolerance.
 */
export interface LineRangeEdit {
	lineRange: [number, number];
	oldText: string;
	newText: string;
}

interface MatchedEdit {
	editIndex: number;
	matchIndex: number;
	matchLength: number;
	newText: string;
}

export interface AppliedEditsResult {
	baseContent: string;
	newContent: string;
}

/**
 * Find oldText in content, trying exact match first, then fuzzy match.
 * When fuzzy matching is used, the returned contentForReplacement is the
 * fuzzy-normalized version of the content (trailing whitespace stripped,
 * Unicode quotes/dashes normalized to ASCII).
 */
export function fuzzyFindText(content: string, oldText: string): FuzzyMatchResult {
	// Try exact match first
	const exactIndex = content.indexOf(oldText);
	if (exactIndex !== -1) {
		return {
			found: true,
			index: exactIndex,
			matchLength: oldText.length,
			usedFuzzyMatch: false,
			contentForReplacement: content,
		};
	}

	// Try fuzzy match - work entirely in normalized space
	const fuzzyContent = normalizeForFuzzyMatch(content);
	const fuzzyOldText = normalizeForFuzzyMatch(oldText);

	for (const variant of newlineVariants(fuzzyOldText)) {
		if (variant.length === 0) continue;
		const fuzzyIndex = fuzzyContent.indexOf(variant);
		if (fuzzyIndex !== -1) {
			return {
				found: true,
				index: fuzzyIndex,
				matchLength: variant.length,
				usedFuzzyMatch: true,
				contentForReplacement: fuzzyContent,
			};
		}
	}

	// Match line-by-line using trimmed equality (fixes indent/tab drift)
	const trimmedBlocks = findTrimmedLineBlocks(fuzzyContent, fuzzyOldText);
	if (trimmedBlocks.length === 1) {
		const { start, end } = trimmedBlocks[0];
		return {
			found: true,
			index: start,
			matchLength: end - start,
			usedFuzzyMatch: true,
			contentForReplacement: fuzzyContent,
		};
	}

	// Markup / wrapped lines: same tokens but different internal spacing
	if (trimmedBlocks.length === 0) {
		const relaxedBlocks = findRelaxedWhitespaceLineBlocks(fuzzyContent, fuzzyOldText);
		if (relaxedBlocks.length === 1) {
			const { start, end } = relaxedBlocks[0];
			return {
				found: true,
				index: start,
				matchLength: end - start,
				usedFuzzyMatch: true,
				contentForReplacement: fuzzyContent,
			};
		}
	}
	// Last-resort anchor match for long blocks with tiny interior drift.
	if (trimmedBlocks.length === 0 && fuzzyOldText.split("\n").length >= 6) {
		const anchored = findAnchorRangeBlocks(fuzzyContent, fuzzyOldText);
		if (anchored.length === 1) {
			const { start, end } = anchored[0];
			return {
				found: true,
				index: start,
				matchLength: end - start,
				usedFuzzyMatch: true,
				contentForReplacement: fuzzyContent,
			};
		}
	}

	return {
		found: false,
		index: -1,
		matchLength: 0,
		usedFuzzyMatch: false,
		contentForReplacement: content,
	};
}

/** Strip UTF-8 BOM if present, return both the BOM (if any) and the text without it */
export function stripBom(content: string): { bom: string; text: string } {
	return content.startsWith("\uFEFF") ? { bom: "\uFEFF", text: content.slice(1) } : { bom: "", text: content };
}

function countOccurrences(content: string, oldText: string): number {
	const fuzzyContent = normalizeForFuzzyMatch(content);
	const fuzzyOldText = normalizeForFuzzyMatch(oldText);
	let maxLiteral = 0;
	for (const variant of newlineVariants(fuzzyOldText)) {
		if (variant.length === 0) continue;
		const n = fuzzyContent.split(variant).length - 1;
		if (n > maxLiteral) maxLiteral = n;
	}
	if (maxLiteral > 0) {
		return maxLiteral;
	}
	const trimmed = findTrimmedLineBlocks(fuzzyContent, fuzzyOldText).length;
	if (trimmed > 0) {
		return trimmed;
	}
	const relaxed = findRelaxedWhitespaceLineBlocks(fuzzyContent, fuzzyOldText).length;
	if (relaxed > 0) return relaxed;
	return findAnchorRangeBlocks(fuzzyContent, fuzzyOldText).length;
}

function getNotFoundError(path: string, editIndex: number, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(
			`Could not find the exact text in ${path}. The old text must match exactly including all whitespace and newlines.`,
		);
	}
	return new Error(
		`Could not find edits[${editIndex}] in ${path}. The oldText must match exactly including all whitespace and newlines.`,
	);
}

function getDuplicateError(path: string, editIndex: number, totalEdits: number, occurrences: number): Error {
	if (totalEdits === 1) {
		return new Error(
			`Found ${occurrences} occurrences of the text in ${path}. The text must be unique. Please provide more context to make it unique.`,
		);
	}
	return new Error(
		`Found ${occurrences} occurrences of edits[${editIndex}] in ${path}. Each oldText must be unique. Please provide more context to make it unique.`,
	);
}

function getEmptyOldTextError(path: string, editIndex: number, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(`oldText must not be empty in ${path}.`);
	}
	return new Error(`edits[${editIndex}].oldText must not be empty in ${path}.`);
}

function getNoChangeError(path: string, totalEdits: number): Error {
	if (totalEdits === 1) {
		return new Error(
			`No changes made to ${path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected.`,
		);
	}
	return new Error(`No changes made to ${path}. The replacements produced identical content.`);
}

/**
 * Apply one or more exact-text replacements to LF-normalized content.
 *
 * All edits are matched against the same original content. Replacements are
 * then applied in reverse order so offsets remain stable. If any edit needs
 * fuzzy matching, the operation runs in fuzzy-normalized content space to
 * preserve current single-edit behavior.
 */
export function applyEditsToNormalizedContent(
	normalizedContent: string,
	edits: Edit[],
	path: string,
): AppliedEditsResult {
	const normalizedEdits = edits.map((edit) => ({
		oldText: normalizeToLF(edit.oldText),
		newText: normalizeToLF(edit.newText),
	}));

	for (let i = 0; i < normalizedEdits.length; i++) {
		if (normalizedEdits[i].oldText.length === 0) {
			throw getEmptyOldTextError(path, i, normalizedEdits.length);
		}
	}

	const initialMatches = normalizedEdits.map((edit) => fuzzyFindText(normalizedContent, edit.oldText));
	const baseContent = initialMatches.some((match) => match.usedFuzzyMatch)
		? normalizeForFuzzyMatch(normalizedContent)
		: normalizedContent;

	const matchedEdits: MatchedEdit[] = [];
	for (let i = 0; i < normalizedEdits.length; i++) {
		const edit = normalizedEdits[i];
		const matchResult = fuzzyFindText(baseContent, edit.oldText);
		if (!matchResult.found) {
			const fc = normalizeForFuzzyMatch(baseContent);
			const fo = normalizeForFuzzyMatch(edit.oldText);
			let literalOcc = 0;
			for (const v of newlineVariants(fo)) {
				if (v.length > 0) literalOcc = Math.max(literalOcc, fc.split(v).length - 1);
			}
			if (literalOcc === 0) {
				const trimmedOcc = findTrimmedLineBlocks(fc, fo).length;
				if (trimmedOcc > 1) {
					throw getDuplicateError(path, i, normalizedEdits.length, trimmedOcc);
				}
				if (trimmedOcc === 0) {
					const relaxedOcc = findRelaxedWhitespaceLineBlocks(fc, fo).length;
					if (relaxedOcc > 1) {
						throw getDuplicateError(path, i, normalizedEdits.length, relaxedOcc);
					}
				}
			}
			throw getNotFoundError(path, i, normalizedEdits.length);
		}

		const occurrences = countOccurrences(baseContent, edit.oldText);
		if (occurrences > 1) {
			throw getDuplicateError(path, i, normalizedEdits.length, occurrences);
		}

		matchedEdits.push({
			editIndex: i,
			matchIndex: matchResult.index,
			matchLength: matchResult.matchLength,
			newText: edit.newText,
		});
	}

	matchedEdits.sort((a, b) => a.matchIndex - b.matchIndex);
	for (let i = 1; i < matchedEdits.length; i++) {
		const previous = matchedEdits[i - 1];
		const current = matchedEdits[i];
		if (previous.matchIndex + previous.matchLength > current.matchIndex) {
			throw new Error(
				`edits[${previous.editIndex}] and edits[${current.editIndex}] overlap in ${path}. Merge them into one edit or target disjoint regions.`,
			);
		}
	}

	let newContent = baseContent;
	for (let i = matchedEdits.length - 1; i >= 0; i--) {
		const edit = matchedEdits[i];
		newContent =
			newContent.substring(0, edit.matchIndex) +
			edit.newText +
			newContent.substring(edit.matchIndex + edit.matchLength);
	}

	if (baseContent === newContent) {
		throw getNoChangeError(path, normalizedEdits.length);
	}

	return { baseContent, newContent };
}

/**
 * Generate a unified diff string with line numbers and context.
 * Returns both the diff string and the first changed line number (in the new file).
 */
export function generateDiffString(
	oldContent: string,
	newContent: string,
	contextLines = 4,
): { diff: string; firstChangedLine: number | undefined } {
	const parts = Diff.diffLines(oldContent, newContent);
	const output: string[] = [];

	const oldLines = oldContent.split("\n");
	const newLines = newContent.split("\n");
	const maxLineNum = Math.max(oldLines.length, newLines.length);
	const lineNumWidth = String(maxLineNum).length;

	let oldLineNum = 1;
	let newLineNum = 1;
	let lastWasChange = false;
	let firstChangedLine: number | undefined;

	for (let i = 0; i < parts.length; i++) {
		const part = parts[i];
		const raw = part.value.split("\n");
		if (raw[raw.length - 1] === "") {
			raw.pop();
		}

		if (part.added || part.removed) {
			// Capture the first changed line (in the new file)
			if (firstChangedLine === undefined) {
				firstChangedLine = newLineNum;
			}

			// Show the change
			for (const line of raw) {
				if (part.added) {
					const lineNum = String(newLineNum).padStart(lineNumWidth, " ");
					output.push(`+${lineNum} ${line}`);
					newLineNum++;
				} else {
					// removed
					const lineNum = String(oldLineNum).padStart(lineNumWidth, " ");
					output.push(`-${lineNum} ${line}`);
					oldLineNum++;
				}
			}
			lastWasChange = true;
		} else {
			// Context lines - only show a few before/after changes
			const nextPartIsChange = i < parts.length - 1 && (parts[i + 1].added || parts[i + 1].removed);
			const hasLeadingChange = lastWasChange;
			const hasTrailingChange = nextPartIsChange;

			if (hasLeadingChange && hasTrailingChange) {
				if (raw.length <= contextLines * 2) {
					for (const line of raw) {
						const lineNum = String(oldLineNum).padStart(lineNumWidth, " ");
						output.push(` ${lineNum} ${line}`);
						oldLineNum++;
						newLineNum++;
					}
				} else {
					const leadingLines = raw.slice(0, contextLines);
					const trailingLines = raw.slice(raw.length - contextLines);
					const skippedLines = raw.length - leadingLines.length - trailingLines.length;

					for (const line of leadingLines) {
						const lineNum = String(oldLineNum).padStart(lineNumWidth, " ");
						output.push(` ${lineNum} ${line}`);
						oldLineNum++;
						newLineNum++;
					}

					output.push(` ${"".padStart(lineNumWidth, " ")} ...`);
					oldLineNum += skippedLines;
					newLineNum += skippedLines;

					for (const line of trailingLines) {
						const lineNum = String(oldLineNum).padStart(lineNumWidth, " ");
						output.push(` ${lineNum} ${line}`);
						oldLineNum++;
						newLineNum++;
					}
				}
			} else if (hasLeadingChange) {
				const shownLines = raw.slice(0, contextLines);
				const skippedLines = raw.length - shownLines.length;

				for (const line of shownLines) {
					const lineNum = String(oldLineNum).padStart(lineNumWidth, " ");
					output.push(` ${lineNum} ${line}`);
					oldLineNum++;
					newLineNum++;
				}

				if (skippedLines > 0) {
					output.push(` ${"".padStart(lineNumWidth, " ")} ...`);
					oldLineNum += skippedLines;
					newLineNum += skippedLines;
				}
			} else if (hasTrailingChange) {
				const skippedLines = Math.max(0, raw.length - contextLines);
				if (skippedLines > 0) {
					output.push(` ${"".padStart(lineNumWidth, " ")} ...`);
					oldLineNum += skippedLines;
					newLineNum += skippedLines;
				}

				for (const line of raw.slice(skippedLines)) {
					const lineNum = String(oldLineNum).padStart(lineNumWidth, " ");
					output.push(` ${lineNum} ${line}`);
					oldLineNum++;
					newLineNum++;
				}
			} else {
				// Skip these context lines entirely
				oldLineNum += raw.length;
				newLineNum += raw.length;
			}

			lastWasChange = false;
		}
	}

	return { diff: output.join("\n"), firstChangedLine };
}

export interface EditDiffResult {
	diff: string;
	firstChangedLine: number | undefined;
}

export interface EditDiffError {
	error: string;
}

/**
 * Compute the diff for one or more edit operations without applying them.
 * Used for preview rendering in the TUI before the tool executes.
 */
export async function computeEditsDiff(
	path: string,
	edits: Edit[],
	cwd: string,
): Promise<EditDiffResult | EditDiffError> {
	const resolvedPath = resolveWorkspacePath(path, cwd, { kind: "file", basenameFallback: true });
	const absolutePath = resolveToCwd(resolvedPath, cwd);

	try {
		// Check if file exists and is readable
		try {
			await access(absolutePath, constants.R_OK);
		} catch {
			return { error: `File not found: ${resolvedPath}` };
		}

		// Read the file
		const rawContent = await readFile(absolutePath, "utf-8");

		// Strip BOM before matching (LLM won't include invisible BOM in oldText)
		const { text: content } = stripBom(rawContent);
		const normalizedContent = normalizeToLF(content);
		const { baseContent, newContent } = applyEditsToNormalizedContent(normalizedContent, edits, resolvedPath);

		// Generate the diff
		return generateDiffString(baseContent, newContent);
	} catch (err) {
		return { error: err instanceof Error ? err.message : String(err) };
	}
}

/**
 * Compute the diff for a single edit operation without applying it.
 * Kept as a convenience wrapper for single-edit callers.
 */
export async function computeEditDiff(
	path: string,
	oldText: string,
	newText: string,
	cwd: string,
): Promise<EditDiffResult | EditDiffError> {
	return computeEditsDiff(path, [{ oldText, newText }], cwd);
}

/**
 * Project a string to lowercase ASCII alphanumerics ([a-z0-9]).
 * Everything else (whitespace, punctuation, quotes, operators, case, Unicode) is dropped.
 * Used for the tolerant `oldText` verification: trust the line range and only
 * check that the model's `oldText` plausibly refers to the targeted lines.
 */
function lowerAlnumProjection(s: string): string {
	return s.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

/**
 * Compare the actual lines at a range with the model-provided `oldText`.
 *
 * Policy: TRUST the line range. `oldText` is only a very flexible guard —
 * we project both sides to lowercase-alphanumeric characters only and check
 * that the oldText projection is a substring of the actual projection (or
 * vice versa for very short lines). That tolerates all whitespace, case,
 * punctuation, quotes, comment styling, and Unicode drift.
 *
 * Returns ok=true when either projection is empty (fully trusts the range)
 * or when one projection contains the other.
 */
function verifyOldTextAgainstLines(
	actualLines: string[],
	expectedOldText: string,
): { ok: boolean; actual: string } {
	const actualJoined = actualLines.join("\n");
	const expectedProj = lowerAlnumProjection(expectedOldText);
	const actualProj = lowerAlnumProjection(actualJoined);

	if (expectedProj.length === 0 || actualProj.length === 0) {
		return { ok: true, actual: actualJoined };
	}
	if (actualProj.includes(expectedProj) || expectedProj.includes(actualProj)) {
		return { ok: true, actual: actualJoined };
	}
	return { ok: false, actual: actualJoined };
}

/**
 * Apply one or more line-range replacements to LF-normalized content.
 *
 * Policy: TRUST the model-provided line numbers. We silently clamp any
 * start/end line that falls outside the file (startLine >= totalLines is
 * treated as "append at end", endLine > totalLines-1 is clamped). We do not
 * enforce non-overlapping ranges; overlapping edits are applied in reverse
 * order of startLine and later-added content simply wins. The only soft
 * guard is the tolerant lowercase-alphanumeric `oldText` check per entry.
 */
export function applyLineRangeEditsToNormalizedContent(
	normalizedContent: string,
	edits: LineRangeEdit[],
	path: string,
): AppliedEditsResult {
	const lines = normalizedContent.split("\n");
	const totalLines = lines.length;

	// Normalize each edit into a safe clamped range. Append-at-end is supported
	// by passing startLine >= totalLines (range becomes [totalLines, totalLines-1],
	// which splice treats as pure insertion at the end).
	type ClampedEdit = {
		start: number;
		end: number; // exclusive splice delete-end (end - start = deleteCount)
		newText: string;
		oldText: string;
		idx: number;
	};
	const clamped: ClampedEdit[] = [];
	for (let i = 0; i < edits.length; i++) {
		const e = edits[i];
		const rawFirst = Array.isArray(e.lineRange) && Number.isInteger(e.lineRange[0]) ? e.lineRange[0] : 0;
		const rawLast = Array.isArray(e.lineRange) && Number.isInteger(e.lineRange[1]) ? e.lineRange[1] : rawFirst;
		const first = Math.max(0, rawFirst);
		// If the model addresses content past the end, treat as "append at end".
		let start = Math.min(first, totalLines);
		let endExclusive: number;
		if (start >= totalLines) {
			start = totalLines;
			endExclusive = totalLines; // pure insertion
		} else {
			const last = Math.max(start, Math.min(rawLast, totalLines - 1));
			endExclusive = last + 1;
		}
		clamped.push({
			start,
			end: endExclusive,
			newText: typeof e.newText === "string" ? e.newText : "",
			oldText: typeof e.oldText === "string" ? e.oldText : "",
			idx: i,
		});
	}

	// Soft oldText verification (the only remaining guard).
	for (const c of clamped) {
		if (c.end <= c.start) continue; // pure insertion — no content to verify
		const slice = lines.slice(c.start, c.end);
		const check = verifyOldTextAgainstLines(slice, c.oldText);
		if (!check.ok) {
			const prefix = clamped.length === 1 ? "" : `edits[${c.idx}] `;
			throw new Error(
				`${prefix}oldText does not match the actual content of lines ${c.start}-${c.end - 1} in ${path}.\n` +
					`--- Your oldText ---\n${c.oldText}\n` +
					`--- Actual file content at lines ${c.start}-${c.end - 1} ---\n${check.actual}`,
			);
		}
	}

	// Apply in reverse order of start so earlier indices stay stable.
	const newLines = lines.slice();
	const applyOrder = clamped.slice().sort((a, b) => b.start - a.start || b.end - a.end);
	for (const c of applyOrder) {
		const replacement = normalizeToLF(c.newText).split("\n");
		newLines.splice(c.start, c.end - c.start, ...replacement);
	}

	const newContent = newLines.join("\n");
	return { baseContent: normalizedContent, newContent };
}
