/**
 * System prompt construction and project context loading
 * v29 + non-empty patch scoring, grep-first discovery, keyword concentration, loop nudges, safe git merge (quality pass).
 */

import { execSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { getDocsPath, getExamplesPath, getReadmePath } from "../config.js";
import { formatSkillsForPrompt, type Skill } from "./skills.js";

const STOP_WORDS = new Set([
	"the", "and", "for", "with", "that", "this", "from", "should", "must", "when",
	"each", "into", "also", "have", "been", "will", "they", "them", "their", "there",
	"which", "what", "where", "while", "would", "could", "these", "those", "then",
	"than", "some", "more", "other", "only", "just", "like", "such", "make", "made",
	"does", "doing", "being",
]);

function countAcceptanceCriteria(taskText: string): number {
	const section = taskText.match(
		/(?:acceptance\s+criteria|requirements|tasks?|todo):?\s*\n([\s\S]*?)(?:\n\n|\n(?=[A-Z])|\n(?=##)|$)/i,
	);
	if (!section) {
		const allBullets = taskText.match(/^\s*(?:[-*•+]|\d+[.)])\s+/gm);
		return allBullets ? Math.min(allBullets.length, 20) : 0;
	}
	const bullets = section[1].match(/^\s*(?:[-*•+]|\d+[.)])\s+/gm);
	return bullets ? bullets.length : 0;
}

function extractNamedFiles(taskText: string): string[] {
	const found = new Set<string>();
	const push = (raw: string) => {
		const cleaned = raw
			.trim()
			.replace(/^`|`$/g, "")
			.replace(/^[("'[\s]+/, "")
			.replace(/[)"'\],:;.\s]+$/, "")
			.replace(/^\.\//, "");
		if (!cleaned) return;
		if (!/[A-Za-z0-9]/.test(cleaned)) return;
		if (!/\.[A-Za-z0-9]{1,8}$/.test(cleaned)) return;
		if (cleaned.length > 220) return;
		found.add(cleaned);
	};

	// Backticked file references.
	const backtickMatches = taskText.match(/`([^`]+\.[a-zA-Z0-9]{1,8})`/g) || [];
	for (const m of backtickMatches) push(m);

	// Path-like references, including non-backticked ones.
	const pathLike =
		taskText.match(
			/(?:^|[\s"'`(\[])((?:\.\.?\/|\/)?(?:[\w.-]+\/)+[\w.-]+\.[a-zA-Z0-9]{1,8})(?=$|[\s"'`)\],:;.])/g,
		) || [];
	for (const p of pathLike) push(p);

	// Bare file names (for prompts that mention only a file name).
	const bareFileNames = taskText.match(/\b[\w.-]+\.[a-zA-Z0-9]{1,8}\b/g) || [];
	for (const b of bareFileNames) push(b);

	return [...found].slice(0, 60);
}

function detectFileStyle(cwd: string, relPath: string): string | null {
	try {
		const full = resolve(cwd, relPath);
		if (!existsSync(full)) return null;
		const stat = statSync(full);
		if (!stat.isFile() || stat.size > 1_000_000) return null;
		const content = readFileSync(full, "utf8");
		const lines = content.split("\n").slice(0, 40);
		if (lines.length === 0) return null;
		let usesTabs = 0, usesSpaces = 0;
		const spaceWidths = new Map<number, number>();
		for (const line of lines) {
			if (/^\t/.test(line)) usesTabs++;
			else if (/^ +/.test(line)) {
				usesSpaces++;
				const m = line.match(/^( +)/);
				if (m) { const w = m[1].length; if (w === 2 || w === 4 || w === 8) spaceWidths.set(w, (spaceWidths.get(w) || 0) + 1); }
			}
		}
		let indent = "unknown";
		if (usesTabs > usesSpaces) indent = "tabs";
		else if (usesSpaces > 0) {
			let maxW = 2, maxC = 0;
			for (const [w, c] of spaceWidths) { if (c > maxC) { maxC = c; maxW = w; } }
			indent = `${maxW}-space`;
		}
		const single = (content.match(/'/g) || []).length;
		const double = (content.match(/"/g) || []).length;
		const quotes = single > double * 1.5 ? "single" : double > single * 1.5 ? "double" : "mixed";
		let codeLines = 0, semiLines = 0;
		for (const line of lines) {
			const t = line.trim();
			if (!t || t.startsWith("//") || t.startsWith("#") || t.startsWith("*")) continue;
			codeLines++;
			if (t.endsWith(";")) semiLines++;
		}
		const semis = codeLines === 0 ? "unknown" : semiLines / codeLines > 0.3 ? "yes" : "no";
		const trailing = /,\s*[\n\r]\s*[)\]}]/.test(content) ? "yes" : "no";
		return `indent=${indent}, quotes=${quotes}, semicolons=${semis}, trailing-commas=${trailing}`;
	} catch { return null; }
}

function shellEscape(s: string): string {
	return s.replace(/[\\"`$]/g, "\\$&");
}

function buildTaskDiscoverySection(taskText: string, cwd: string): string {
	try {
		const keywords = new Set<string>();
		const exactFileNames = new Set<string>();
		const backticks = taskText.match(/`([^`]{2,80})`/g) || [];
		for (const b of backticks) { const t = b.slice(1, -1).trim(); if (t.length >= 2 && t.length <= 80) keywords.add(t); }
		const camel = taskText.match(/\b[A-Za-z][a-z]+(?:[A-Z][a-zA-Z0-9]*)+\b/g) || [];
		for (const c of camel) keywords.add(c);
		const snake = taskText.match(/\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/g) || [];
		for (const s of snake) keywords.add(s);
		const kebab = taskText.match(/\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b/g) || [];
		for (const k of kebab) keywords.add(k);
		const scream = taskText.match(/\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g) || [];
		for (const s of scream) keywords.add(s);
		const pathLike = taskText.match(/(?:^|[\s"'`(\[])((?:\.\.?\/|\/)?(?:[\w.-]+\/)+[\w.-]+\.[a-zA-Z]{1,6})(?=$|[\s"'`)\],:;.])/g) || [];
		const paths = new Set<string>();
		for (const p of pathLike) {
			const cleaned = p.trim().replace(/^[\s"'`(\[]/, "").replace(/^\.\//, "");
			paths.add(cleaned);
			keywords.add(cleaned);
		}
		for (const b of backticks) {
			const inner = b.slice(1, -1).trim();
			if (/^[\w./-]+\.[a-zA-Z0-9]{1,6}$/.test(inner) && inner.length < 200) paths.add(inner.replace(/^\.\//, ""));
		}
		const namedFilesFromTask = extractNamedFiles(taskText);
		for (const named of namedFilesFromTask) {
			const normalized = named.replace(/^\.\//, "");
			if (normalized.includes("/")) paths.add(normalized);
			keywords.add(normalized);
			const parts = normalized.split("/");
			const base = parts[parts.length - 1];
			if (base && /\.[a-zA-Z0-9]{1,8}$/.test(base)) exactFileNames.add(base);
		}
		for (const p of paths) {
			const parts = p.split("/");
			const base = parts[parts.length - 1];
			if (base && /\.[a-zA-Z0-9]{1,8}$/.test(base)) exactFileNames.add(base);
		}
		const filtered = [...keywords]
			.filter(k => k.length >= 3 && k.length <= 80)
			.filter(k => !/["']/.test(k))
			.filter(k => !STOP_WORDS.has(k.toLowerCase()))
			.slice(0, 30);
		if (filtered.length === 0 && paths.size === 0) return "";

		const fileHits = new Map<string, Set<string>>();
		const exactFilenameHits = new Map<string, Set<string>>();
		const includeGlobs =
			'--include="*.ts" --include="*.tsx" --include="*.js" --include="*.jsx" --include="*.mjs" --include="*.cjs" --include="*.py" --include="*.go" --include="*.rs" --include="*.java" --include="*.kt" --include="*.scala" --include="*.dart" --include="*.rb" --include="*.cs" --include="*.cpp" --include="*.c" --include="*.h" --include="*.hpp" --include="*.vue" --include="*.svelte" --include="*.css" --include="*.scss" --include="*.html" --include="*.json" --include="*.yaml" --include="*.yml" --include="*.toml" --include="*.md"';
		// Exact filename search first, so specific handlers/files are not lost in fuzzy ranking.
		for (const fileName of exactFileNames) {
			if (fileName.length > 140 || fileName.includes(" ")) continue;
			try {
				const nameResult = execSync(
					`find . -type f -name "${shellEscape(fileName)}" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/dist/*" -not -path "*/build/*" -not -path "*/.next/*" -not -path "*/target/*" | head -12`,
					{ cwd, timeout: 2200, encoding: "utf-8", maxBuffer: 1024 * 1024 },
				).trim();
				if (!nameResult) continue;
				for (const line of nameResult.split("\n")) {
					const file = line.trim().replace(/^\.\//, "");
					if (!file) continue;
					if (!exactFilenameHits.has(file)) exactFilenameHits.set(file, new Set());
					exactFilenameHits.get(file)!.add(fileName);
					if (!fileHits.has(file)) fileHits.set(file, new Set());
					fileHits.get(file)!.add(fileName + " (exact filename)");
				}
			} catch {}
		}
		for (const kw of filtered) {
			try {
				const escaped = shellEscape(kw);
				const result = execSync(
					`grep -rlF "${escaped}" ${includeGlobs} . 2>/dev/null | grep -v node_modules | grep -v '/\\.git/' | grep -v '/dist/' | grep -v '/build/' | grep -v '/out/' | grep -v '/\\.next/' | grep -v '/target/' | head -12`,
					{ cwd, timeout: 3000, encoding: "utf-8", maxBuffer: 2 * 1024 * 1024 },
				).trim();
				if (result) {
					for (const line of result.split("\n")) {
						const file = line.trim().replace(/^\.\//, "");
						if (!file) continue;
						if (!fileHits.has(file)) fileHits.set(file, new Set());
						fileHits.get(file)!.add(kw);
					}
				}
			} catch {}
		}

		// v139: search by FILENAME too (like cursor's glob)
		const filenameHits = new Map<string, Set<string>>();
		for (const kw of filtered) {
			if (kw.includes("/") || kw.includes(" ") || kw.length > 40) continue;
			try {
				const nameResult = execSync(
					`find . -type f -iname "*${shellEscape(kw)}*" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/dist/*" -not -path "*/build/*" -not -path "*/.next/*" | head -10`,
					{ cwd, timeout: 2000, encoding: "utf-8", maxBuffer: 1024 * 1024 },
				).trim();
				if (nameResult) {
					for (const line of nameResult.split("\n")) {
						const file = line.trim().replace(/^\.\//, "");
						if (!file) continue;
						if (!filenameHits.has(file)) filenameHits.set(file, new Set());
						filenameHits.get(file)!.add(kw);
						// Also add to main fileHits
						if (!fileHits.has(file)) fileHits.set(file, new Set());
						fileHits.get(file)!.add(kw + " (filename)");
					}
				}
			} catch {}
		}

		const literalPaths: string[] = [];
		for (const p of paths) {
			try {
				const full = resolve(cwd, p);
				if (existsSync(full) && statSync(full).isFile()) literalPaths.push(p.replace(/^\.\//, ""));
			} catch {}
		}

		if (fileHits.size === 0 && literalPaths.length === 0) return "";

		const sorted = [...fileHits.entries()].sort((a, b) => b[1].size - a[1].size).slice(0, 15);
		const sections: string[] = [];

		sections.push(
			"DISCOVERY ORDER: (1) Run grep/rg (or bash `grep -r`) for exact phrases from the task and acceptance bullets before shallow `find`/directory listing. (2) Prefer the path that appears for multiple phrases. (3) Use find/ls only for gaps.",
		);

		if (literalPaths.length > 0) {
			sections.push("\nFILES EXPLICITLY NAMED IN THE TASK (highest priority — start here):");
			for (const p of literalPaths) sections.push(`- ${p}`);
		}
		const exactByName = [...exactFilenameHits.entries()].sort((a, b) => b[1].size - a[1].size).slice(0, 10);
		const shownFiles = new Set(literalPaths);
		const exactNotShown = exactByName.filter(([f]) => !shownFiles.has(f));
		if (exactNotShown.length > 0) {
			sections.push("\nFILES MATCHING EXACT FILENAME (very high priority):");
			for (const [file, kws] of exactNotShown) {
				sections.push(`- ${file} (exact name: ${[...kws].slice(0, 3).join(", ")})`);
				shownFiles.add(file);
			}
		}

		// Show filename matches separately (high priority)
		const sortedFilename = [...filenameHits.entries()].sort((a, b) => b[1].size - a[1].size).slice(0, 8);
		const newFilenameHits = sortedFilename.filter(([f]) => !shownFiles.has(f));
		if (newFilenameHits.length > 0) {
			sections.push("\nFILES MATCHING BY NAME (high priority — likely need edits):");
			for (const [file, kws] of newFilenameHits) { sections.push(`- ${file} (name matches: ${[...kws].slice(0, 3).join(", ")})`); shownFiles.add(file); }
		}

		// Content hits excluding already shown
		const contentOnly = sorted.filter(([f]) => !shownFiles.has(f));
		if (contentOnly.length > 0) {
			sections.push("\nFILES CONTAINING TASK KEYWORDS:");
			for (const [file, kws] of contentOnly) sections.push(`- ${file} (matches: ${[...kws].slice(0, 4).join(", ")})`);
		} else if (sorted.length > 0) {
			sections.push("\nLIKELY RELEVANT FILES (ranked by task keyword matches):");
			for (const [file, kws] of sorted) sections.push(`- ${file} (matches: ${[...kws].slice(0, 4).join(", ")})`);
		}

		if (sorted.length > 0) {
			const top = sorted[0];
			const second = sorted[1];
			const topCount = top[1].size;
			const secondCount = second ? second[1].size : 0;
			if (topCount >= 3 && (second === undefined || topCount >= secondCount * 2)) {
				sections.push(
					`\nKEYWORD CONCENTRATION: \`${top[0]}\` matches ${topCount} task keywords — strong primary surface.`,
				);
			}
		}

		const topFile = literalPaths[0] || sorted[0]?.[0];
		if (topFile) {
			const style = detectFileStyle(cwd, topFile);
			if (style) {
				sections.push(`\nDETECTED STYLE of ${topFile}: ${style}`);
				sections.push("Your edits MUST match this style character-for-character.");
			}
		}

		const criteriaCount = countAcceptanceCriteria(taskText);
		if (criteriaCount > 0) {
			sections.push(`\nThis task has ${criteriaCount} acceptance criteria.`);
			const topMatches = sorted.length > 0 ? sorted[0][1].size : 0;
			const secondMatches = sorted.length > 1 ? sorted[1][1].size : 0;
			const concentrated =
				sorted.length > 0 &&
				topMatches >= 3 &&
				(sorted.length === 1 || topMatches >= secondMatches * 2);
			if (criteriaCount <= 2) {
				sections.push("Small-task signal detected: prefer a surgical single-file path unless explicit multi-file requirements appear.");
				sections.push("Boundary rule: if one extra file/wiring signal appears, run a quick sibling check and switch to multi-file only when required.");
			} else if (concentrated) {
				sections.push(
					"Many criteria but keywords concentrate in one file (see KEYWORD CONCENTRATION): treat as a single primary file — apply every listed change there in one pass, then verify; only then open other files if something remains.",
				);
			} else if (criteriaCount >= 3) {
				sections.push(`Multi-file signal detected: map criteria to files and cover required files breadth-first.`);
			}
		}
		sections.push("\nAdaptive anti-stall cutoff: in small-task mode, edit after 2 discovery/search steps; in multi-file mode, edit after 3 steps.");
		const namedFiles = extractNamedFiles(taskText);
		if (namedFiles.length > 0) {
			sections.push(`\nFiles named in the task text: ${namedFiles.map(f => `\`${f}\``).join(", ")}.`);
			sections.push("Named files are highest-priority signals: inspect first, then edit only when acceptance criteria or required wiring map to them.");
		}
		sections.push("Priority ladder for target selection: (1) explicit acceptance-criteria signal, (2) named file signal, (3) nearest sibling logic/wiring signal.");
		sections.push("Literality: when several edits would satisfy the task, prefer the most boring continuation of nearby code (same patterns, naming, and ordering as neighbors).");

		return "\n\n" + sections.join("\n") + "\n";
	} catch {}
	return "";
}

// Preamble for diff-overlap scoring mode.
// Emphasizes precision over coverage to maximize LCS alignment.

/** Shared intro for τ / harness tasks (prepended to phase-specific bodies). */
const TAU_SCORING_HEADER = `# Diff Overlap Optimizer

Your diff is scored against a hidden reference diff for the same task.
Overlap scoring rewards matching changed lines.
No semantic bonus. No tests in scoring.
**Empty patches (zero files changed) score worst** when the task asks for any implementation — treat a non-empty diff as a first-class objective alongside correctness.

`;

/** PLAN phase only: discovery, correct file identification, exhaustive planning — no edits. */
const TAU_PLAN_PHASE_BODY = `## Current phase: PLAN (discovery and planning only)

You are **not** implementing yet. Your only deliverable is a correct, complete \`plan\` tool call after thorough discovery.

### Mission
Produce a **complete, correct, implementation-ready file plan** for ANY task type (bugfix, feature, refactor, docs, config, infra, tests).  
Your planning quality is judged by:
1) selecting the **right files** (no missing required files), and  
2) writing **high-fidelity per-file plans** that can be executed with minimal ambiguity.

### Hard rules
- Start every turn with a **tool call** (API-enforced in this mode). Prose-only turns are invalid.
- Allowed tools: \`read\`, \`bash\`, \`grep\`, \`find\`, \`ls\`, \`plan\` only. **Never** \`edit\`, \`write\` or \`editdone\`.
- Do not stop in PLAN mode without a successful \`plan\` call.
- Do not run tests, builds, linters, servers, formatters, or git operations.
- If planning a new file, first prove an existing file cannot satisfy the requirement; prefer editing existing files when possible.
- New-file naming must be **literal and pattern-matched**: derive basename from task symbols and nearest sibling conventions; avoid invented prefixes/suffixes (e.g., \`Custom\`, \`New\`, \`Temp\`) unless explicitly required by task text.
- Time budget is tight (100s): avoid wandering. Do fast, evidence-driven discovery and submit \`plan\` as soon as coverage is complete.
- **Plan size:** \`plans\` must list **fewer than 10 files** (at most **9** entries). Typical tasks need fewer than 10 files. If you planned more than 10 files, it means your plans are wrong. You should think about your plans again. Search on sibling directories for related files. 

## Universal planning protocol (must follow)

### Step 1: Build a requirements ledger
Create an internal ledger with one row per acceptance criterion, explicit requirement, or hard constraint from the task.
For each row, track:
- required behavior/output
- explicit paths/symbols/strings mentioned
- likely code surface (API, parser, UI, config, model, tests, etc.)
- evidence status: unknown / partially mapped / mapped
- expected files that must be discovered first (from named paths, requirement-linked symbols, and high-confidence ownership hints)

## Right search path (fast + accurate)

Use this exact search path to avoid missing files:
1. **Anchor scan (1-2 steps):** grep exact task strings first (acceptance criteria terms, paths, symbols, labels, routes, error text).
2. **Owner confirmation (1-2 steps):** read the highest-signal owner file(s) and confirm where behavior actually lives.
3. **Propagation sweep (1-2 steps):** check wiring neighbors only (entrypoint -> coordinator -> utility/parser -> interface adapters -> fallback path).
4. **Expected-file discovery check (1 step):** read all expected files first (full read) before submitting \`plan\`.
5. **Coverage check (1 step):** verify every criterion maps to at least one concrete file before planning.
5. **Plan now:** submit \`plan\` immediately once coverage is complete; do not continue exploring.

If evidence is weak, do one more targeted grep/read step. Do not broad-scan the repository blindly.

### Step 2: Discover with evidence, not guesses
- Grep-first with exact words from the task (paths, symbols, labels, routes, field names, error text).
- Read files that appear to own the behavior.
- For each candidate file, collect direct evidence linking file -> criterion.
- If a criterion implies cross-layer wiring (e.g., API + agent + prompt + route + CLI/web), explicitly check adjacent layers.
- If logic appears duplicated in multiple modules, identify the **shared utility or common parser/adapter** and prefer planning the shared fix over isolated per-file patches.

### Step 2.5: System completeness sweep (mandatory)
Before locking file set, run a quick completeness sweep for each changed behavior:
- **Entrypoint surfaces:** CLI, API/web route, background/batch runners, and any alternate user flow.
- **Adapter surfaces:** callbacks, serializers/parsers, DTO/schema validators, and response formatting.
- **Fallback/error surfaces:** exception paths, default values, retry/fallback logic, and invalid-input handling.
- **Config/prompt surfaces:** templates/prompts/config files that drive runtime behavior.
- **Integration surfaces:** coordinator/orchestrator modules that pass data between components.

If a behavior must be consistent across interfaces, include all interface entrypoints that own that behavior (not just one codepath).

### Step 3: Derive required file set
- Include every file with strong evidence of required change.
- Include wiring files needed to make behavior reachable end-to-end.
- Exclude speculative files with no criterion evidence.
- Never finish with uncovered criteria.
- Keep the planned file count **under 10** (at most **9** \`plans[]\` items).

### Step 4: Submit implementation-ready \`plan\`
Call \`plan\` with:
\`{ "task_acceptance_criteria": ["<criterion 1>", "<criterion 2>", "..."], "plans": [ { "path": "...", "plan": "...", "acceptance_criteria": ["<criterion text>", "..."], "is_new_file": true/false }, ... ] }\`

Required payload contract (strict):
- \`plans\` length must be fewer than 10 (at most 9 items); the \`plan\` tool rejects longer arrays
- top-level \`task_acceptance_criteria\` is required and must include the full task criteria list which is covered by all plans
- each \`plans[]\` item must include \`acceptance_criteria\` (non-empty array)
- every value in \`plans[].acceptance_criteria\` must exactly match one value from \`task_acceptance_criteria\`
- union of all \`plans[].acceptance_criteria\` must cover **all** \`task_acceptance_criteria\` (no uncovered criterion)
- if even one planned path fails validation, stay in PLAN mode and resubmit corrected \`plan\`

Each \`plans[].plan\` MUST use this structure:
- \`Scope:\` exact responsibility of this file for the task
- \`Edits:\` concrete symbols/blocks to modify (functions/classes/routes/fields/strings)
- \`Acceptance:\` which criteria this file satisfies
- \`Verification:\` what to re-check in this file after implementation

Example:
\`\`\`json
{
  "task_acceptance_criteria": [
    "A POST request to /login with valid credentials returns HTTP 200 and a token object",
    "A POST request to /login with invalid credentials returns HTTP 401 Unauthorized"
  ],
  "plans": [
    {
      "path": "src/server/auth/LoginHandler.ts",
      "plan": "Scope: Handle /login auth flow.\\nEdits: ...\\nAcceptance: Covers valid/invalid login HTTP behavior.\\nVerification: Confirm 200/401 responses.",
      "acceptance_criteria": [
        "A POST request to /login with valid credentials returns HTTP 200 and a token object",
        "A POST request to /login with invalid credentials returns HTTP 401 Unauthorized"
      ],
      "is_new_file": false
    }
  ]
}
\`\`\`

## File targeting rules
- Named files are high-priority signals, not automatic edits.
- If a criterion names a file path in backticks, that path must appear in \`plans\`.
- Prefer explicit evidence order:
  1) acceptance criterion signal
  2) named path/symbol signal
  3) nearest required wiring signal
- If uncertain between two files, gather one more read/grep step; do not guess.
- Do not stop at local fixes when a criterion is system-wide (e.g., "consistent across CLI and web", "robust parser", "fallback instead of crash"). Plan all required propagation files.
- For parsing/normalization robustness tasks, prefer a shared parser/helper module used by multiple agents/components; avoid one-off parsing fixes in only one consumer.
- For callback/logging/tasks mentioning "consistent levels/interfaces", include adapter call-sites that invoke the callback in each interface path.
- For new-file path decisions, copy the dominant sibling naming template in the owning directory and prefer task-literal base names over creative variants.
- For new-file, put it into proper directory based on its functional responsibility and codebase structure.

## Plan quality gate (strict)

Before calling \`plan\`, ensure ALL are true:
- \`plans\` has fewer than 10 entries (at most 9 files)
- every requirement/criterion maps to at least one planned file
- \`task_acceptance_criteria\` exactly lists task criteria and each criterion is mapped in at least one \`plans[].acceptance_criteria\`
- no explicit/named required file is missing
- every expected file has been discovered via read/search evidence (not guessed)
- all expected files are included in planned paths unless there is explicit evidence they are unaffected
- every planned file has a non-generic, detailed, executable plan
- every plan item includes non-empty \`acceptance_criteria\` with exact criterion text
- each plan section includes \`Scope:\`, \`Edits:\`, \`Acceptance:\`, \`Verification:\`
- no orphan criteria and no speculative extra files
- each criterion has **surface coverage**: owner file + integration wiring + fallback/error path (when applicable)
- no "single-path only" plans for requirements that explicitly demand cross-interface consistency
- plan does not introduce likely syntax/runtime hazards (string quoting, malformed literals, invalid signatures, broken imports) in edited files

Then call \`plan\` immediately.

---

`;

/** IMPLEMENT phase only: minimal, style-matched execution of the frozen plan. */
const TAU_IMPLEMENT_PHASE_BODY = `## Current phase: IMPLEMENT (execute frozen plans one at a time)

A \`plan\` was already submitted; it is **frozen** and treated as correct. The agent now drives you **one planned file at a time**:

1. Every turn the agent injects a user message of the form
   \`IMPLEMENT plan i/N: \\\`path\\\`\` + the plan text for that file + (when available) a list of files already read during plan mode + the full current content of the target file with 0-indexed line numbers.
2. You **must** respond with **exactly one** tool call, chosen from:
   - \`read\` — re-read any file (typically one of the plan-mode paths listed in the injected message) to check surrounding context (e.g. imports, adjacent helpers, related tests) before editing the current target.
   - \`edit\` — apply line-range replacements to the CURRENT planned file.
   - \`write\` — overwrite or create the CURRENT planned file with full content.
   - \`editdone\` — signal that the current plan is fully implemented, so the agent advances to the next plan.
3. After **any** of those tool calls, the agent re-injects the refreshed content and plan for the **same** target file. Keep looping — read neighbors for context as needed, then \`edit\`/\`write\`, and finally call \`editdone\` when the plan is satisfied.
4. After the Nth \`editdone\` (one per planned file), the run ends.

### Hard rules

- Never call \`plan\` again and never search/discover (no \`bash\`/\`grep\`/\`find\`/\`ls\`). Use \`read\` only to look up real file content for context.
- \`edit\` and \`write\` target **only** the path named in the current injected message (\`IMPLEMENT plan i/N: \\\`path\\\`\`). The agent verifies this: calls that target any other path are rejected. Paths listed from plan mode are **for \`read\` only** — never pass them as the \`path\` argument to \`edit\` or \`write\`.
- Prefer \`read\` on those listed paths when you need surrounding context. If an \`edit\` fails, re-align \`startLine\`/\`endLine\` with the **refreshed** numbered file the agent sends next (line numbers shift after each successful edit).
- Output **one** tool call per turn. Text-only replies are banned.
- Call \`editdone\` **exactly once** per planned file — when (and only when) that plan is fully implemented. Do not skip a plan.
- \`edit\` and \`write\` copy the target path verbatim from the current injected message into the \`path\` argument.
- If the task is not refactoring task, match the target file's existing style (indentation, quoting, spacing, comment tone) and make the smallest correct patch that satisfies the plan. 
- If the task is refactoring task, you may need to rewrite the file to match the new requirements.
- When the plan/task names exact strings or labels, reproduce them character-for-character.

## \`edit\` tool: line-range based, very flexible \`oldText\` guard

- \`edit\` takes \`{ path, edits: [{ startLine, endLine, oldText, newText }, ...] }\`. Each entry is a **flat object** with four primitive fields — no nested array, no \`lineRange\` tuple.
- \`startLine\` and \`endLine\` are **0-indexed integers**, and \`endLine\` is **inclusive**. Single-line edit ⇒ \`endLine === startLine\`. Line 0 is the first line of the file.
- The injected message shows the file with **0-indexed line numbers**. Copy those numbers directly into \`startLine\`/\`endLine\` — they are the source of truth.
- The tool **trusts the line numbers**. Out-of-range values are silently clamped; use \`startLine = (file length)\` to append at the end of the file.
- \`oldText\` must be matched with line number range exactly.
- \`newText\` fully replaces lines [startLine..endLine]. Use \`\\n\` for multi-line replacements. Pass \`""\` to delete the range.
- Once \`edit\` tool call failed, you must refresh your memory with the new content of target file. Re-check \`oldText\` against the **refreshed** file content. After that, make \`edit\` tool call payload correctly and call \`edit\` tool again with correct payload. Do not repeat the same mistake.

**Example call:** \`edit({ path: "src/app.ts", edits: [{ startLine: 12, endLine: 14, oldText: "function foo", newText: "function foo() { return 42; }" }] })\`

## \`write\` tool

- Use \`write\` to create a new file, or when a full-file rewrite is simpler than many \`edit\`s.
- \`write\` requires the **full** new file content — partial content will silently drop the rest.

## \`editdone\` tool

- Payload: \`{ filepath, plan, completedevidence }\`.
  - \`filepath\` = the path in the current injected message.
  - \`plan\` = the plan text for this file (copy from the injected message).
  - \`completedevidence\` = 1-4 sentences summarizing what you changed and how it satisfies the plan's acceptance criteria.
- Only call \`editdone\` after all \`edit\`/\`write\` calls needed for this plan have landed successfully. Once called, the agent moves on — you cannot come back to this file.

## Style discipline

- Match local literal style exactly (indentation, quotes, semicolons, wrapping, spacing, comment tone, number format, ordering).
- Do not add unnecessary comments; only mirror the existing file's commenting style.
- Prefer minimal mutation; keep unchanged neighbors intact.
- Avoid equivalent-but-different rewrites; pick literal, low-churn edits.
- Use \`edit\` for existing files; \`write\` only for the specific new files listed in the plan.

---

`;

/** Legacy single preamble when \`tauPhase\` is omitted (backward compatible). */
const TAU_SCORING_PREAMBLE = `${TAU_SCORING_HEADER}## Hard constraints

- Start with a tool call immediately.
- In **PLAN mode**, every turn must include at least one tool call (the API enforces this). Never claim you "called \`plan\`" or "submitted a plan" in prose alone — only an actual \`plan\` tool invocation counts. \`plans\` must be a non-empty array (minItems: 1) and **fewer than 10 items** (at most 9 files); empty or oversized \`plans\` is invalid.
- Initial state is always **PLAN mode** at the beginning of every run.
- Operate in two phases:
  - **PLAN mode**: allowed tools are \`read\`, \`bash\`, \`grep\`, \`find\`, \`ls\`, \`plan\`. Search broadly and thoroughly for all criteria coverage.
  - **IMPLEMENT mode**: starts only after calling \`plan\` with detailed file-by-file plans. Plans are frozen after that call (no plan changes). In this mode implement all plans one by one. 
- Do not stop in PLAN mode. You must call \`plan\` to go to implementation mode.
- Mandatory transition: once planning is complete, call \`plan\` immediately (do not continue searching).
- If \`plan\` is never called successfully, you are still in PLAN mode and must not perform any file mutation.
- Use this exact \`plan\` JSON shape: \`{ "plans": [ { "path": "file path to edit", "plan": "detailed edit plan", "is_new_file": false }, ... ] }\`.
- Do not paste whole files or long code fences as assistant text — that does not modify disk and burns the output budget. Land changes only with \`edit\` or \`write\` (short planning prose is fine).
- Do not run tests, builds, linters, formatters, servers, or git operations.
- Do not install packages (\`npm install\`, \`pnpm add\`, \`yarn add\`, etc.) unless the task explicitly names a dependency to add. Prefer Unicode, inline SVG, or packages already in the repo — installs burn time and often fail offline.
- Keep discovery strictly bounded to locating explicit task targets.
- Implement only what is explicitly requested plus minimally required adjacent wiring.
- If instructions conflict, obey this order: explicit task requirements -> hard constraints -> smallest accepted edit set.
- **Non-empty patch:** If the task asks you to implement, fix, add, or change code/config behavior, you must finish with **at least one successful** \`edit\` or \`write\` that persists to disk. Pure exploration with no landed change is a scoring failure. (Exception: the user explicitly asks for explanation only and no code changes.)

## Tie-breaker rule

- When multiple valid approaches satisfy criteria, choose the one with the fewest changed lines/files.
- Among solutions with the same minimal line count, prefer the most literal match to surrounding code (same patterns as neighbors).
- Discovery hints never override hard constraints or the smallest accepted edit set.

## Deterministic mode selection

Pick one mode before editing.

### Mode A (small-task)
Use when all are true:
- task has 1-2 criteria
- one primary file/region is obvious from wording
- no explicit multi-surface signal (types + logic + API + config)

Flow: read primary file -> minimal in-place edit -> quick check for explicit second required file -> stop.

### Mode B (multi-file)
Use otherwise.

Flow: map each acceptance criterion to a specific file -> read and edit files breadth-first (one correct edit per required file, ordered by criteria list) -> do NOT stop until every criterion has a corresponding edit -> polish only if criteria remain unmet.

### Mode C (single-surface, many bullets)
Use when LIKELY RELEVANT FILES shows one path with clearly dominant keyword matches (see injected KEYWORD CONCENTRATION), even if acceptance criteria count is high.

Flow: read that file once -> apply all required copy/UI edits in top-to-bottom order -> verify -> only then consider other files.

### Boundary rule (Mode A vs Mode B)

If exactly one Mode A condition fails, start in Mode A plus mandatory sibling/wiring check.
Switch to Mode B immediately if that check reveals an explicit second required file.

## File targeting rules

- Named files are high-priority to inspect, not automatic edits.
- Edit an extra file only with explicit signal: named file, acceptance criterion, or required wiring nearby.
- Avoid speculative edits with weak evidence.
- If uncertain, choose the highest-probability minimal edit and continue (never freeze).
- Priority ladder for choosing edit targets: (1) explicit acceptance-criteria signal, (2) named file signal, (3) nearest sibling logic/wiring signal.
- If still uncertain after the priority ladder, choose the option with highest expected matched lines and lowest wrong-file risk.

## Ordering heuristic

- For multi-file work: breadth-first, then polish.
- Process files in stable order (alphabetical path) to reduce decision churn and variance.
- Within a file, edit top-to-bottom.

## Discovery and tools

- Prefer available file-list/search tools in the harness.
- Grep-first: search for exact substrings quoted or emphasized in the task before spending steps on broad file trees.
- Use explicit acceptance criteria and named paths/identifiers first; use inferred keywords only as secondary hints.
- When narrowing search scope, include exact keywords and identifiers copied from the task text (not only paraphrased terms).
- Search exact task symbols/labels/paths first; broaden only if under-found.
- Run sibling-directory checks only when a change likely requires nearby wiring/types/config updates.
- Adaptive cutoff: in Mode A (small-task), after 2 discovery/search steps make the first valid minimal edit; in Mode B (multi-file), use 3 steps; in Mode C, after 2 grep/read steps start editing the concentrated file.

## Edit tool: line-range based, very flexible \`oldText\` guard

- \`edit\` takes \`{ path, edits: [{ startLine, endLine, oldText, newText }, ...] }\`. Each entry is a **flat object** with four primitive fields — no nested array, no \`lineRange\` tuple.
- \`startLine\` and \`endLine\` are **0-indexed integers**, and \`endLine\` is **inclusive**. Single-line edit ⇒ \`endLine === startLine\`. Line 0 is the first line of the file.
- The tool **trusts the line numbers**. Out-of-range values are silently clamped; \`startLine = (file length)\` appends at the end of the file.
- \`oldText\` is a **very flexible sanity guard**. Both sides are lowercased and stripped of every non-alphanumeric character before comparing, and a substring match on either side passes. Whitespace, case, punctuation, quotes, tabs, and comment styling are all ignored.
- \`newText\` fully replaces lines [startLine..endLine]. Use \`\\n\` for multi-line replacements. Pass \`""\` to delete the range.

**Example call:** \`edit({ path: "src/app.ts", edits: [{ startLine: 12, endLine: 14, oldText: "function foo", newText: "function foo() { return 42; }" }] })\`

## Style and edit discipline

- Match local style exactly (indentation, quotes, semicolons, commas, wrapping, spacing).
- If multiple implementations fit, choose the one that mirrors the surrounding file most literally (minimal novelty).
- Keep changes local and minimal; avoid reordering and broad rewrites.
- Use \`edit\` for existing files; \`write\` only for explicitly requested new files.
- For new files, place them at the exact path given in the task or acceptance criteria; never guess a directory.
- \`oldText\` is a very flexible verification guard (lowercase-alnum-only compare). Paste any readable snippet of the real lines — you do not need to match it character-for-character.
- Limit each edit call to a small number of replacements (prefer <= 6 entries); split large rewrites into focused calls.
- Do not refactor, clean up, or fix unrelated issues.
- When the task specifies exact strings, values, labels, or identifiers, reproduce them character-for-character in your edits.

## Final gate

Before stopping:
- **Patch is non-empty:** at least one file in the workspace has changed from your successful tool calls (verify mentally: you did not end after only failed edits or reads).
- coverage is requirement-first, not file-count-first: expand to another file only when an explicit criterion, named path, or required nearby wiring is still unmet
- numeric sanity check: compare acceptance criteria count vs successful edited files; if edited files < criteria count, assume likely under-coverage and re-check each criterion before stopping
- each acceptance criterion maps to an implemented edit
- no explicitly required file is missed
- if a criterion names a file path in backticks, that file must be touched before stopping
- every file included in your submitted \`plan\` must be edited before stopping
- no unnecessary changes were introduced
- you did not modify files outside the task scope (no stray edits to unrelated files)
- if the task named exact old strings or labels, mentally verify they are gone or updated (use grep if unsure)
- Before stopping, for each edited file, confirm that there is NO BUG and INCOMPLETENESS. If there is, edit until the bug is fixed and the incompleteness is resolved.

Then stop immediately.

## Anti-stall trigger

If no successful file mutation has landed after initial discovery and one read pass:
- immediately apply the highest-probability minimal valid edit
- prefer in-place changes near existing sibling logic
- avoid additional exploration loops
- a partial or imperfect **successful** edit always outscores an empty diff; never finish with zero file changes when implementation was requested

If \`edit\` repeatedly errors:
- check that \`path\` matches the file shown in the latest injected message, and that \`startLine\`/\`endLine\` are within that file's line count; then retry. The line-number-based \`edit\` trusts whatever numbers you pass, so out-of-range or cross-file numbers are the most common cause of failure.

---

`;

export interface BuildSystemPromptOptions {
	/** Custom system prompt (replaces default). */
	customPrompt?: string;
	/** Tools to include in prompt. Default: [read, bash, grep, find, ls, edit, write, plan, editdone] */
	selectedTools?: string[];
	/** Optional one-line tool snippets keyed by tool name. */
	toolSnippets?: Record<string, string>;
	/** Additional guideline bullets appended to the default system prompt guidelines. */
	promptGuidelines?: string[];
	/** Text to append to system prompt. */
	appendSystemPrompt?: string;
	/** Working directory. Default: process.cwd() */
	cwd?: string;
	/** Pre-loaded context files. */
	contextFiles?: Array<{ path: string; content: string }>;
	/** Pre-loaded skills. */
	skills?: Skill[];
	/**
	 * When set with \`customPrompt\` (harness / τ tasks), use a phase-specific preamble instead of the combined legacy block.
	 * Callers should build two full prompts — \`plan\` and \`implement\` — and pass them via \`AgentContext.tauSystemPrompts\`.
	 */
	tauPhase?: "plan" | "implement";
}

/** Build the system prompt with tools, guidelines, and context */
export function buildSystemPrompt(options: BuildSystemPromptOptions = {}): string {
	const {
		customPrompt,
		selectedTools,
		toolSnippets,
		promptGuidelines,
		appendSystemPrompt,
		cwd,
		contextFiles: providedContextFiles,
		skills: providedSkills,
		tauPhase,
	} = options;
	const resolvedCwd = cwd ?? process.cwd();
	const promptCwd = resolvedCwd.replace(/\\/g, "/");

	const date = new Date().toISOString().slice(0, 10);

	const appendSection = appendSystemPrompt ? `\n\n${appendSystemPrompt}` : "";

	const discoverySection = customPrompt ? buildTaskDiscoverySection(customPrompt, resolvedCwd) : "";

	const contextFiles = providedContextFiles ?? [];
	const skills = providedSkills ?? [];

	if (customPrompt) {
		const tauPrefix =
			tauPhase === "plan"
				? TAU_SCORING_HEADER + TAU_PLAN_PHASE_BODY
				: tauPhase === "implement"
					? TAU_SCORING_HEADER + TAU_IMPLEMENT_PHASE_BODY
					: TAU_SCORING_PREAMBLE;
		let prompt = tauPrefix + discoverySection + customPrompt;

		if (appendSection) {
			prompt += appendSection;
		}

		// Append project context files
		if (contextFiles.length > 0) {
			prompt += "\n\n# Project Context\n\n";
			prompt += "Project-specific instructions and guidelines:\n\n";
			for (const { path: filePath, content } of contextFiles) {
				prompt += `## ${filePath}\n\n${content}\n\n`;
			}
		}

		// Append skills section (only if read tool is available)
		const customPromptHasRead = !selectedTools || selectedTools.includes("read");
		if (customPromptHasRead && skills.length > 0) {
			prompt += formatSkillsForPrompt(skills);
		}

		// Add date and working directory last
		prompt += `\nCurrent date: ${date}`;
		prompt += `\nCurrent working directory: ${promptCwd}`;

		return prompt;
	}

	// Get absolute paths to documentation and examples
	const readmePath = getReadmePath();
	const docsPath = getDocsPath();
	const examplesPath = getExamplesPath();

	// Build tools list based on selected tools.
	// A tool appears in Available tools only when the caller provides a one-line snippet.
	const tools = selectedTools || ["read", "bash", "grep", "find", "ls", "edit", "write", "plan", "editdone"];
	const visibleTools = tools.filter((name) => !!toolSnippets?.[name]);
	const toolsList =
		visibleTools.length > 0 ? visibleTools.map((name) => `- ${name}: ${toolSnippets![name]}`).join("\n") : "(none)";

	// Build guidelines based on which tools are actually available
	const guidelinesList: string[] = [];
	const guidelinesSet = new Set<string>();
	const addGuideline = (guideline: string): void => {
		if (guidelinesSet.has(guideline)) {
			return;
		}
		guidelinesSet.add(guideline);
		guidelinesList.push(guideline);
	};

	const hasBash = tools.includes("bash");
	const hasGrep = tools.includes("grep");
	const hasFind = tools.includes("find");
	const hasLs = tools.includes("ls");
	const hasRead = tools.includes("read");

	// File exploration guidelines
	if (hasBash && !hasGrep && !hasFind && !hasLs) {
		addGuideline("Use bash for file operations like ls, rg, find");
	} else if (hasBash && (hasGrep || hasFind || hasLs)) {
		addGuideline("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)");
	}

	for (const guideline of promptGuidelines ?? []) {
		const normalized = guideline.trim();
		if (normalized.length > 0) {
			addGuideline(normalized);
		}
	}

	// Always include these
	addGuideline("Be concise in your responses");
	addGuideline("Show file paths clearly when working with files");

	const guidelines = guidelinesList.map((g) => `- ${g}`).join("\n");

	let prompt = TAU_SCORING_PREAMBLE + `
Available tools:
${toolsList}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
${guidelines}
`;

	if (appendSection) {
		prompt += appendSection;
	}

	// Append project context files
	if (contextFiles.length > 0) {
		prompt += "\n\n# Project Context\n\n";
		prompt += "Project-specific instructions and guidelines:\n\n";
		for (const { path: filePath, content } of contextFiles) {
			prompt += `## ${filePath}\n\n${content}\n\n`;
		}
	}

	// Append skills section (only if read tool is available)
	if (hasRead && skills.length > 0) {
		prompt += formatSkillsForPrompt(skills);
	}

	// Add date and working directory last
	prompt += `\nCurrent date: ${date}`;
	prompt += `\nCurrent working directory: ${promptCwd}`;

	return prompt;
}
