import { accessSync, constants, readdirSync, statSync } from "node:fs";
import * as os from "node:os";
import type { Dirent } from "node:fs";
import { basename, isAbsolute, join, relative, resolve as resolvePath, sep } from "node:path";
import { URL } from "node:url";

const UNICODE_SPACES = /[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g;
const NARROW_NO_BREAK_SPACE = "\u202F";
function normalizeUnicodeSpaces(str: string): string {
	return str.replace(UNICODE_SPACES, " ");
}

function tryMacOSScreenshotPath(filePath: string): string {
	return filePath.replace(/ (AM|PM)\./g, `${NARROW_NO_BREAK_SPACE}$1.`);
}

function tryNFDVariant(filePath: string): string {
	// macOS stores filenames in NFD (decomposed) form, try converting user input to NFD
	return filePath.normalize("NFD");
}

function tryCurlyQuoteVariant(filePath: string): string {
	// macOS uses U+2019 (right single quotation mark) in screenshot names like "Capture d'écran"
	// Users typically type U+0027 (straight apostrophe)
	return filePath.replace(/'/g, "\u2019");
}

function fileExists(filePath: string): boolean {
	try {
		accessSync(filePath, constants.F_OK);
		return true;
	} catch {
		return false;
	}
}

function normalizeAtPrefix(filePath: string): string {
	return filePath.startsWith("@") ? filePath.slice(1) : filePath;
}

const QUOTED_PATH_RE = /^(['"`])(.*)\1$/s;

/** Normalize common model path formatting mistakes before resolving. */
export function normalizePathInput(filePath: string): string {
	let normalized = normalizeUnicodeSpaces(normalizeAtPrefix(filePath)).trim();
	const quoted = normalized.match(QUOTED_PATH_RE);
	if (quoted) {
		normalized = quoted[2].trim();
	}
	if (normalized.startsWith("file:")) {
		try {
			const u = new URL(normalized);
			normalized = decodeURIComponent(u.pathname || normalized);
		} catch {
			/* keep original when URL parsing fails */
		}
	}
	if (normalized.startsWith("@/")) {
		normalized = normalized.slice(2);
	}
	normalized = normalized.replace(/\\/g, "/");
	if (normalized.startsWith("./")) {
		normalized = normalized.slice(2);
	}
	return normalized;
}

export function expandPath(filePath: string): string {
	const normalized = normalizePathInput(filePath);
	if (normalized === "~") {
		return os.homedir();
	}
	if (normalized.startsWith("~/")) {
		return os.homedir() + normalized.slice(1);
	}
	return normalized;
}

/**
 * Resolve a path relative to the given cwd.
 * Handles ~ expansion and absolute paths.
 */
export function resolveToCwd(filePath: string, cwd: string): string {
	const expanded = expandPath(filePath);
	if (isAbsolute(expanded)) {
		return expanded;
	}
	return resolvePath(cwd, expanded);
}

function isPathInsideRoot(root: string, candidate: string): boolean {
	let rootAbs = resolvePath(root);
	let candAbs = resolvePath(candidate);
	if (process.platform === "win32") {
		rootAbs = rootAbs.toLowerCase();
		candAbs = candAbs.toLowerCase();
	}
	if (candAbs === rootAbs) return true;
	const prefix = rootAbs.endsWith(sep) ? rootAbs : rootAbs + sep;
	return candAbs.startsWith(prefix);
}

function toRepoRelativePosixPath(root: string, absolutePath: string): string {
	const rel = relative(root, absolutePath);
	if (!rel || rel === "." || rel === "") return ".";
	return rel.split(sep).join("/");
}

const SKIP_WALK_DIRS = new Set([
	"node_modules",
	".git",
	".next",
	"dist",
	"build",
	"coverage",
	".cache",
	".turbo",
	"__pycache__",
	".venv",
	"venv",
]);

function findByExactBasename(root: string, fileName: string, maxMatches = 32): string[] {
	if (!fileName || fileName === "." || fileName === "..") return [];
	const out: string[] = [];
	const queue: string[] = [root];
	let visited = 0;
	const MAX_NODES = 100_000;
	while (queue.length > 0 && out.length < maxMatches && visited < MAX_NODES) {
		const dir = queue.pop()!;
		let entries: Dirent[];
		try {
			entries = readdirSync(dir, { withFileTypes: true });
		} catch {
			continue;
		}
		for (const ent of entries) {
			visited++;
			if (ent.isSymbolicLink()) continue;
			const full = join(dir, ent.name);
			if (ent.isDirectory()) {
				if (SKIP_WALK_DIRS.has(ent.name)) continue;
				queue.push(full);
				continue;
			}
			if (ent.isFile() && ent.name === fileName) {
				out.push(toRepoRelativePosixPath(root, full));
				if (out.length >= maxMatches) break;
			}
		}
	}
	return out;
}

function rankBasenameMatches(original: string, matches: string[]): string {
	if (matches.length === 1) return matches[0];
	const needle = original.replace(/\\/g, "/").replace(/^\.?\//, "").replace(/\/$/, "");
	const segments = needle.split("/").filter(Boolean);
	const score = (candidate: string): number => {
		const c = candidate.replace(/\\/g, "/").replace(/^\.?\//, "");
		let s = 0;
		if (c === needle || c.endsWith("/" + needle) || c.endsWith(needle)) s += 10_000;
		for (let i = 1; i <= Math.min(segments.length, 8); i++) {
			const suf = segments.slice(-i).join("/");
			if (suf && c.endsWith(suf)) s += i * 100;
		}
		return s;
	};
	return [...matches].sort((a, b) => score(b) - score(a) || a.localeCompare(b))[0];
}

export type ResolveWorkspacePathOptions = {
	/** Expected node kind; if mismatched, behaves as unresolved. */
	kind?: "file" | "directory" | "any";
	/** Search workspace by exact basename when target doesn't exist. */
	basenameFallback?: boolean;
};

/** Resolve and sanitize model paths with optional basename fallback inside cwd. */
export function resolveWorkspacePath(filePath: string, cwd: string, options: ResolveWorkspacePathOptions = {}): string {
	const normalized = normalizePathInput(filePath);
	const absolute = resolveToCwd(normalized, cwd);
	if (!isPathInsideRoot(cwd, absolute)) {
		return normalized;
	}
	const kind = options.kind ?? "any";
	if (fileExists(absolute)) {
		try {
			accessSync(absolute, constants.F_OK);
			const stat = statSync(absolute);
			const kindOk =
				kind === "any" || (kind === "file" && stat.isFile()) || (kind === "directory" && stat.isDirectory());
			if (kindOk) {
				return toRepoRelativePosixPath(cwd, absolute);
			}
		} catch {
			return normalized;
		}
	}
	if (!options.basenameFallback || kind === "directory") {
		return normalized;
	}
	const matches = findByExactBasename(cwd, basename(normalized.replace(/\/$/, "")));
	if (matches.length === 0) return normalized;
	return rankBasenameMatches(normalized, matches);
}

export function resolveReadPath(filePath: string, cwd: string): string {
	const resolved = resolveToCwd(resolveWorkspacePath(filePath, cwd, { kind: "file", basenameFallback: true }), cwd);

	if (fileExists(resolved)) {
		return resolved;
	}

	// Try macOS AM/PM variant (narrow no-break space before AM/PM)
	const amPmVariant = tryMacOSScreenshotPath(resolved);
	if (amPmVariant !== resolved && fileExists(amPmVariant)) {
		return amPmVariant;
	}

	// Try NFD variant (macOS stores filenames in NFD form)
	const nfdVariant = tryNFDVariant(resolved);
	if (nfdVariant !== resolved && fileExists(nfdVariant)) {
		return nfdVariant;
	}

	// Try curly quote variant (macOS uses U+2019 in screenshot names)
	const curlyVariant = tryCurlyQuoteVariant(resolved);
	if (curlyVariant !== resolved && fileExists(curlyVariant)) {
		return curlyVariant;
	}

	// Try combined NFD + curly quote (for French macOS screenshots like "Capture d'écran")
	const nfdCurlyVariant = tryCurlyQuoteVariant(nfdVariant);
	if (nfdCurlyVariant !== resolved && fileExists(nfdCurlyVariant)) {
		return nfdCurlyVariant;
	}

	return resolved;
}
