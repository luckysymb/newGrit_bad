import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "@sinclair/typebox";
import { spawnSync } from "node:child_process";
import { basename } from "node:path";
import type { ToolDefinition } from "../extensions/types.js";
import { resolveToCwd, resolveWorkspacePath } from "./path-utils.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";

const MIN_PLAN_CHARS = 280;
const MIN_PLAN_LINES = 4;
const REQUIRED_PLAN_SECTIONS = ["Edits", "Readrequired", "Verification"] as const;
/** Real tasks almost always touch under 16 files; cap keeps plans from fragmenting into dozens of micro-edits. */
const MAX_PLAN_ITEMS = 15;
const planItemSchema = Type.Object(
	{
		path: Type.String({ description: "File path to edit in implementation mode" }),
		plan: Type.String({ description: "Detailed edit plan for this file" }),
		acceptance_criteria: Type.Array(Type.String(), {
			description:
				"Task acceptance criteria covered by this file plan. Each value must exactly match one item in top-level task_acceptance_criteria.",
			minItems: 1,
		}),
		is_new_file: Type.Boolean({
			description: "Whether this plan item creates a new file. Use false for existing files that must be edited.",
		}),
	},
	{ additionalProperties: false },
);

const planSchema = Type.Object(
	{
		task_acceptance_criteria: Type.Array(Type.String(), {
			description: "Exact task acceptance criteria list for this task.",
			minItems: 1,
		}),
		plans: Type.Array(planItemSchema, {
			description:
				`Implementation plan items (each: target file, detailed edit instructions, is_new_file). Must be fewer than 16 entries — at most ${MAX_PLAN_ITEMS} files. If you planned more than 16 files, it means your plans are wrong. You should think about your plans again. Search on sibling directories for related files.`,
			minItems: 1,
			maxItems: MAX_PLAN_ITEMS,
		}),
		plan_mode_read_paths: Type.Optional(Type.Array(Type.String(), {
			description:
				"Optional helper paths from PLAN-mode reads. Used only by path guard to prefer exact-filename matches from already-read files.",
		})),
	},
	{ additionalProperties: false },
);

export type PlanToolInput = Static<typeof planSchema>;

type NormalizedPlanItem = {
	path: string;
	plan: string;
	acceptance_criteria: string[];
	is_new_file: boolean;
};

type PlanValidationResult = {
	path: string;
	is_new_file: boolean;
	validation_result: "passed" | "failed";
	message: string;
	suggested_paths?: string[];
};


export interface PlanToolDetails {
	planCount: number;
	paths: string[];
	allPassed: boolean;
	criteriaAllCovered: boolean;
	uncoveredCriteria: string[];
	validationResults: PlanValidationResult[];
}

function normalizePlanArgs(input: unknown): PlanToolInput {
	// Accept both:
	// 1) { plans: [{path, plan}, ...] }
	// 2) [{path, plan}, ...]
	if (Array.isArray(input)) {
		return { plans: input as any[] } as PlanToolInput;
	}
	if (!input || typeof input !== "object") return input as PlanToolInput;
	const obj = input as Record<string, unknown>;
	if (Array.isArray(obj.plans)) return input as PlanToolInput;
	return input as PlanToolInput;
}

function validatePlanDetailText(planText: string): { ok: boolean; reason?: string } {
	const text = planText.trim();
	if (text.length < MIN_PLAN_CHARS) {
		return {
			ok: false,
			reason: `failed: plan too short (${text.length} chars). Minimum is ${MIN_PLAN_CHARS} chars`,
		};
	}
	const nonEmptyLines = text
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter((line) => line.length > 0);
	if (nonEmptyLines.length < MIN_PLAN_LINES) {
		return {
			ok: false,
			reason: `failed: plan too shallow (${nonEmptyLines.length} non-empty lines). Minimum is ${MIN_PLAN_LINES}`,
		};
	}
	const missingSections = REQUIRED_PLAN_SECTIONS.filter((section) => {
		const re = new RegExp(`\\b${section}\\s*:`, "i");
		return !re.test(text);
	});
	if (missingSections.length > 0) {
		return {
			ok: false,
			reason: `failed: missing required sections (${missingSections.join(", ")}). Required format: Edits:, Readrequired:, Verification:`,
		};
	}
	const editBullets = extractSectionBullets(text, "Edits");
	if (editBullets.length === 0) {
		return {
			ok: false,
			reason: "failed: Edits section must contain at least one `- ...` bullet",
		};
	}
	if (editBullets.some((b) => b.length < 40)) {
		return {
			ok: false,
			reason:
				"failed: each Edits bullet must be detailed/self-contained (>= 40 chars).",
		};
	}
	const readrequiredBullets = extractSectionBullets(text, "Readrequired");
	if (readrequiredBullets.length === 0) {
		return {
			ok: false,
			reason: "failed: Readrequired section must contain at least one `- <path>` bullet",
		};
	}
	const dedup = new Set(readrequiredBullets.map((p) => normalizeRepoRelativePath(p)));
	if (dedup.size !== readrequiredBullets.length) {
		return {
			ok: false,
			reason: "failed: Readrequired contains duplicate paths; list each required file exactly once",
		};
	}
	return { ok: true };
}

function addMissingRequiredSections(planText: string, acceptanceCriteria: string[]): string {
	let text = planText.trim();
	const hasSection = (name: string): boolean => new RegExp(`\\b${name}\\s*:`, "i").test(text);
	if (!hasSection("Edits")) {
		text += "\n\nEdits:\n- Update concrete symbols/blocks required by the task (name exact functions, fields, routes, literals, and before/after behavior).";
	}
	if (!hasSection("Readrequired")) {
		text += "\n\nReadrequired:\n- <exact/path/from/discovery/that/was/read>";
	}
	if (!hasSection("Verification")) {
		const crit = acceptanceCriteria.length > 0 ? acceptanceCriteria.join(" | ") : "Task criteria mapped to this file plan.";
		text += `\n\nVerification: Re-read this file and confirm all planned edits are present and consistent for criteria: ${crit}`;
	}
	return text;
}

function extractSectionBullets(planText: string, sectionName: string): string[] {
	const m = new RegExp(`\\b${sectionName}\\s*:`, "i").exec(planText);
	if (!m) return [];
	const rest = planText.slice(m.index + m[0].length);
	const nextSection = /\n\s*[A-Za-z][A-Za-z0-9_ -]*\s*:/.exec(rest);
	const block = nextSection ? rest.slice(0, nextSection.index) : rest;
	return block
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter((line) => line.startsWith("- "))
		.map((line) => line.slice(2).trim())
		.filter((line) => line.length > 0);
}

export function createPlanToolDefinition(): ToolDefinition<typeof planSchema, PlanToolDetails> {
	return {
		name: "plan",
		label: "plan",
		description:
			`Submit PLAN->IMPLEMENT handoff JSON. Agent handshake: the **first** assistant turn that contains **only** this \`plan\` tool is a **draft** — the agent echoes your payload with a self-audit prompt and **does not** run validation yet. If you call **any** other tool on a later turn, that resets the handshake. The **second** assistant turn that contains **only** \`plan\` again runs full validation and may transition to IMPLEMENT. Required interface: { \"task_acceptance_criteria\": [\"...\"], \"plans\": [{ \"path\": \"relative/file/path\", \"plan\": \"Edits:\\n- ... detailed bullet ...\\nReadrequired:\\n- exact/path/from/discovery/that/was/read\\nVerification: ...\", \"acceptance_criteria\": [\"criterion covered by this plan item\"], \"is_new_file\": false }] }. \`plans\` must be non-empty and have fewer than 16 items (at most ${MAX_PLAN_ITEMS}); each item must include all keys. Readrequired is mandatory and exhaustive per item (must include ALL needed read files, including the item's own path). The order of \`plans\` is load-bearing: implement mode iterates them sequentially and drops per-plan context after each one finishes, so submit them in dependency order (new leaf files first, consumers next, wiring/registration last).`,
		promptSnippet:
			"Call plan with exact JSON interface: { task_acceptance_criteria: [...], plans: [{ path, plan, acceptance_criteria, is_new_file }] }. Order plans in dependency order — leaf files first, wiring last.",
		promptGuidelines: [
			"Two-step handshake: first `plan`-only turn = draft echo (no validation). Any turn with a non-`plan` tool resets the handshake. Second consecutive `plan`-only turn = validated commit to IMPLEMENT.",
			"Call plan only after broad and thorough exploration in plan mode",
			"When you make plan for a file, read that file first and then make the plan",
			`Submit fewer than 16 plan entries (at most ${MAX_PLAN_ITEMS} files). If you planned more than 16 files, it means your plans are wrong and you searched only a few files. Search more files on sibling directories for related files.`,
			"Include every target file needed to satisfy all acceptance criteria",
			"Use exact interface keys: task_acceptance_criteria, path, plan, acceptance_criteria, is_new_file (no aliases)",
			"Each plan item must declare which acceptance criteria it covers via acceptance_criteria",
			"Coverage validation is count-based: the number of task_acceptance_criteria must match criteria covered across plan items",
			"When creating new files, use literal task/symbol names and nearest sibling naming patterns; avoid invented prefixes/suffixes unless explicitly requested",
			"Each plan item must be implementation-ready and include sections: Edits:, Readrequired:, Verification: .",
			"Under **Edits:** use one `- …` bullet per discrete change; each bullet must be fully self-contained with concrete symbols/steps so it is understandable.",
			"Under **Readrequired:** list all file paths needed to complete this item; each path must be an exact verbatim path already read in PLAN mode and copied from discovery tools' output.",
			"Readrequired must be exhaustive per plan item: include ALL files you must read during IMPLEMENT for that item (owner file + required helpers/types/config/wiring/import targets). Missing any needed file in Readrequired is a hard plan defect.",
			"Set is_new_file=false for existing files and true only for files that must be newly created (basename must not already exist anywhere in the repo)",
			"Order plans in DEPENDENCY ORDER (bottom-up / leaf-first): plans[] is executed sequentially and the agent drops per-plan context after each editdone, so later plans only see files on disk after earlier plans landed. Submit new leaf files (interfaces, DTOs, helpers) first, logic that uses them next, and wiring/registration/delegation last. If plan B references a symbol introduced by plan A, A must come before B.",
			"Self-check ordering: for every prefix plans[0..N-1], every symbol referenced by those plans must already exist (either in the pre-existing repo or introduced by one of those earlier plans). If not, reorder.",
		],
		parameters: planSchema,
		prepareArguments: normalizePlanArgs,
		async execute(_toolCallId, input: PlanToolInput) {
			const cwd = process.cwd();
			const planModeReadPaths = Array.isArray((input as any).plan_mode_read_paths)
				? [...new Set(
					((input as any).plan_mode_read_paths as unknown[])
						.map((p) => (typeof p === "string" ? p.trim() : ""))
						.filter((p) => p.length > 0),
				)]
				: [];
			const normalizedTaskCriteria: string[] = [...new Set(
				((input.task_acceptance_criteria as string[] | undefined) ?? [])
					.map((c: string) => c.trim())
					.filter((c: string) => c.length > 0),
			)];
			if (normalizedTaskCriteria.length === 0) {
				throw new Error("plan requires non-empty task_acceptance_criteria");
			}
			const normalized: NormalizedPlanItem[] = input.plans
				.map((p: { path: string; plan: string; acceptance_criteria: string[]; is_new_file: boolean }) => ({
					path: p.path.trim(),
					plan: p.plan.trim(),
					acceptance_criteria: Array.isArray(p.acceptance_criteria)
						? [...new Set(p.acceptance_criteria.map((c) => (typeof c === "string" ? c.trim() : "")).filter((c) => c.length > 0))]
						: [],
					is_new_file: Boolean(p.is_new_file),
				}))
				.filter((p: NormalizedPlanItem) => p.path.length > 0 && p.plan.length > 0);
			if (normalized.length === 0) {
				throw new Error("plan requires at least one valid { path, plan, is_new_file } item");
			}
			if (normalized.length > MAX_PLAN_ITEMS) {
				throw new Error(
					`plan must list fewer than 16 files: at most ${MAX_PLAN_ITEMS} plan items (got ${normalized.length}). You should think about your plans again. Search on sibling directories for related files.`,
				);
			}
			ensureShellValidationToolsAvailable();
			const normalizedReadPathSet = new Set(
				planModeReadPaths
					.map((p) => normalizeRepoRelativePath(p))
					.filter((p) => p.length > 0),
			);
			const autoNormalizedExistingPaths: string[] = [];
			const autoNormalizedReadPathFixes: Array<{ from: string; to: string }> = [];
			const validationResults: PlanValidationResult[] = normalized.map((item) => {
				const normalizedPlanText = addMissingRequiredSections(item.plan, item.acceptance_criteria);
				if (item.acceptance_criteria.length === 0) {
					return {
						path: item.path,
						is_new_file: item.is_new_file,
						validation_result: "failed",
						message: "failed: acceptance_criteria must be non-empty for each plan item",
					};
				}
				const detailValidation = validatePlanDetailText(normalizedPlanText);
				if (!detailValidation.ok) {
					return {
						path: item.path,
						is_new_file: item.is_new_file,
						validation_result: "failed",
						message: detailValidation.reason ?? "failed: insufficient plan detail",
					};
				}
				const readrequiredPaths = extractSectionBullets(normalizedPlanText, "Readrequired");
				for (const rrPathRaw of readrequiredPaths) {
					let rrPath = rrPathRaw;
					const rrNorm = normalizeRepoRelativePath(rrPath);
					if (!normalizedReadPathSet.has(rrNorm)) {
						const alphaOnlyMatches = findAlphaOnlyFullPathMatchesInReadPaths(planModeReadPaths, rrPath, 20);
						if (alphaOnlyMatches.length === 1) {
							rrPath = alphaOnlyMatches[0];
						} else {
							const flexibleMatches = findFlexibleMatchesInReadPaths(planModeReadPaths, rrPath, 20);
							if (flexibleMatches.length === 1) {
								rrPath = flexibleMatches[0];
							}
						}
					}
					if (!normalizedReadPathSet.has(normalizeRepoRelativePath(rrPath))) {
						return {
							path: item.path,
							is_new_file: item.is_new_file,
							validation_result: "failed",
							message:
								`failed: Readrequired path must be already read in PLAN mode (exact or auto-normalizable from read paths): ${rrPathRaw}`,
						};
					}
					const rrResolved = resolveWorkspacePath(rrPath, cwd, { kind: "file", basenameFallback: false });
					const rrAbs = resolveToCwd(rrResolved, cwd);
					if (!fileExistsViaTest(rrAbs)) {
						return {
							path: item.path,
							is_new_file: item.is_new_file,
							validation_result: "failed",
							message: `failed: Readrequired path does not exist as file: ${rrPathRaw}`,
						};
					}
				}
				const resolved = resolveWorkspacePath(item.path, cwd, { kind: "file", basenameFallback: false });
				const abs = resolveToCwd(resolved, cwd);
				const existingAtExactPath = fileExistsViaTest(abs);
				const effectiveIsNewFile = item.is_new_file && !existingAtExactPath;
				if (item.is_new_file && existingAtExactPath) {
					autoNormalizedExistingPaths.push(item.path);
				}
				if (effectiveIsNewFile) {
					const fileName = basename(item.path.replace(/\/+$/, ""));
					const conflicts = findPathsByBasenameViaFind(cwd, fileName, 32);
					if (conflicts.length > 0) {
						return {
							path: item.path,
							is_new_file: effectiveIsNewFile,
							validation_result: "failed",
							message: "failed: is_new_file but a file with this basename already exists",
							suggested_paths: conflicts,
						};
					}
					return {
						path: item.path,
						is_new_file: effectiveIsNewFile,
						validation_result: "passed",
						message: "passed",
					};
				}
				if (fileExistsViaTest(abs)) {
					return {
						path: item.path,
						is_new_file: effectiveIsNewFile,
						validation_result: "passed",
						message: item.is_new_file && existingAtExactPath
							? "passed: normalized is_new_file=false because path already exists"
							: "passed",
					};
				}
				const fileName = basename(item.path.replace(/\/+$/, ""));
				// FIRST PASS (highest priority): a-z-only full-path normalization
				// against PLAN-mode read paths.
				// If exactly one read-path matches, auto-normalize immediately.
				const alphaOnlyFullPathMatches = findAlphaOnlyFullPathMatchesInReadPaths(planModeReadPaths, item.path, 20);
				if (alphaOnlyFullPathMatches.length === 1) {
					const autoPath = alphaOnlyFullPathMatches[0];
					const autoResolved = resolveWorkspacePath(autoPath, cwd, { kind: "file", basenameFallback: false });
					const autoAbs = resolveToCwd(autoResolved, cwd);
					if (fileExistsViaTest(autoAbs)) {
						autoNormalizedReadPathFixes.push({ from: item.path, to: autoPath });
						item.path = autoPath;
						return {
							path: item.path,
							is_new_file: effectiveIsNewFile,
							validation_result: "passed",
							message: "passed: auto-normalized path from plan_mode_read_paths a-z-only full-path match",
						};
					}
				}
				// Flexible path guard order:
				// 1) Prefer exact filename matches from PLAN-mode read paths.
				//    - exactly one: auto-normalize and pass
				//    - multiple: fail + suggest only those read paths
				// 2) Only if zero, fall back to repo-wide basename search suggestions.
				const readPathMatches = findFlexibleMatchesInReadPaths(planModeReadPaths, item.path, 20);
				if (readPathMatches.length === 1) {
					const autoPath = readPathMatches[0];
					const autoResolved = resolveWorkspacePath(autoPath, cwd, { kind: "file", basenameFallback: false });
					const autoAbs = resolveToCwd(autoResolved, cwd);
					if (fileExistsViaTest(autoAbs)) {
						autoNormalizedReadPathFixes.push({ from: item.path, to: autoPath });
						item.path = autoPath;
						return {
							path: item.path,
							is_new_file: effectiveIsNewFile,
							validation_result: "passed",
							message: "passed: auto-normalized path from plan_mode_read_paths exact filename match",
						};
					}
				}
				if (readPathMatches.length > 1) {
					return {
						path: item.path,
						is_new_file: effectiveIsNewFile,
						validation_result: "failed",
						message: "failed: path not found; multiple exact filename matches found in plan_mode_read_paths",
						suggested_paths: readPathMatches,
					};
				}
				const suggestions = findPathsByBasenameViaFind(cwd, fileName, 20);
				return {
					path: item.path,
					is_new_file: effectiveIsNewFile,
					validation_result: "failed",
					message: suggestions.length > 0 ? "failed: path not found; suggestions available" : "failed: path not found",
					suggested_paths: suggestions,
				};
			});
			const coveredCriteria = new Set<string>();
			for (const item of normalized) {
				for (const c of item.acceptance_criteria) coveredCriteria.add(c);
			}
			const uncoveredCriteriaCount = Math.max(0, normalizedTaskCriteria.length - coveredCriteria.size);
			const criteriaAllCovered = coveredCriteria.size >= normalizedTaskCriteria.length;
			const uncoveredCriteria: string[] = uncoveredCriteriaCount > 0
				? [`count_mismatch: expected ${normalizedTaskCriteria.length}, covered ${coveredCriteria.size}`]
				: [];
			const allPassed = validationResults.every((r) => r.validation_result === "passed") && criteriaAllCovered;
			const reportLines = validationResults.map((r, idx) => {
				if (r.validation_result === "passed") {
					return `#${idx + 1} ${r.path} => passed`;
				}
				const suggestions = r.suggested_paths && r.suggested_paths.length > 0
					? `; suggested_paths=${r.suggested_paths.join(", ")}`
					: "";
				return `#${idx + 1} ${r.path} => failed${suggestions}`;
			});
			const normalizationNote = autoNormalizedExistingPaths.length > 0
				? `Auto-normalization: set is_new_file=false for existing path(s): ${[...new Set(autoNormalizedExistingPaths)].join(", ")}\n`
				: "";
			const readPathFixNote = autoNormalizedReadPathFixes.length > 0
				? `Auto-normalization from plan_mode_read_paths exact filename match:\n${autoNormalizedReadPathFixes
					.map((f) => `- ${f.from} -> ${f.to}`)
					.join("\n")}\n`
				: "";
			return {
				content: [
					{
						type: "text",
						text: allPassed
							? `Plan validation passed for all ${normalized.length} file(s).\n${normalizationNote}${readPathFixNote}Validation_result:\n${reportLines.join("\n")}`
							: `Plan validation failed. Fix failed paths/criteria coverage and call plan again.\nEach plan item must be detailed and include sections: Edits:, Readrequired:, Verification:.\n${normalizationNote}${readPathFixNote}${!criteriaAllCovered ? `Uncovered acceptance criteria: ${uncoveredCriteria.slice(0, 15).join(" | ")}\n` : ""}Validation_result:\n${reportLines.join("\n")}`,
					},
				],
				details: {
					planCount: normalized.length,
					paths: normalized.map((p: NormalizedPlanItem) => p.path),
					allPassed,
					criteriaAllCovered,
					uncoveredCriteria,
					validationResults,
				},
			};
		},
		renderCall(args, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			const count = Array.isArray((args as any)?.plans) ? (args as any).plans.length : 0;
			text.setText(`${theme.fg("toolTitle", theme.bold("plan"))} ${theme.fg("accent", `${count} item(s)`)}`);
			return text;
		},
		renderResult(result, _options, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			const output =
				result.content
					?.filter((c: any) => c.type === "text")
					.map((c: any) => c.text || "")
					.join("\n") ?? "";
			text.setText(output ? `\n${theme.fg("toolOutput", output)}` : "");
			return text;
		},
	};
}

export function createPlanTool(): AgentTool<typeof planSchema> {
	return wrapToolDefinition(createPlanToolDefinition());
}

export const planToolDefinition = createPlanToolDefinition();
export const planTool = createPlanTool();

let _shellToolsChecked = false;

function spawnErrorIsENOENT(err: unknown): boolean {
	return typeof err === "object" && err !== null && "code" in err && (err as { code: unknown }).code === "ENOENT";
}

function ensureShellValidationToolsAvailable(): void {
	if (_shellToolsChecked) return;
	const findProbe = spawnSync("find", [".", "-maxdepth", "0"], { encoding: "utf8" });
	if (findProbe.error && spawnErrorIsENOENT(findProbe.error)) {
		throw new Error("plan tool validation requires `find` on PATH");
	}
	const testProbe = spawnSync("test", ["-d", process.cwd()], { encoding: "utf8" });
	if (testProbe.error && spawnErrorIsENOENT(testProbe.error)) {
		throw new Error("plan tool validation requires `test` on PATH");
	}
	_shellToolsChecked = true;
}

/** Existence check via `test -f` (no Node fs). */
function fileExistsViaTest(absolutePath: string): boolean {
	const r = spawnSync("test", ["-f", absolutePath], { encoding: "utf8" });
	return r.status === 0;
}

/**
 * List repo-relative paths to regular files whose basename equals `fileName`, via `find`.
 * Mirrors typical harness ignores (node_modules, .git, dist, …).
 */
function findPathsByBasenameViaFind(repoRoot: string, fileName: string, maxMatches: number): string[] {
	if (!fileName || fileName === "." || fileName === "..") return [];
	const args = [
		".",
		"-type",
		"f",
		"-name",
		fileName,
		"!",
		"-path",
		"*/node_modules/*",
		"!",
		"-path",
		"*/.git/*",
		"!",
		"-path",
		"*/dist/*",
		"!",
		"-path",
		"*/build/*",
		"!",
		"-path",
		"*/.next/*",
		"!",
		"-path",
		"*/coverage/*",
		"!",
		"-path",
		"*/.venv/*",
		"!",
		"-path",
		"*/venv/*",
		"!",
		"-path",
		"*/__pycache__/*",
		"!",
		"-path",
		"*/.turbo/*",
		"!",
		"-path",
		"*/.cache/*",
	];
	const r = spawnSync("find", args, {
		cwd: repoRoot,
		encoding: "utf8",
		maxBuffer: 12 * 1024 * 1024,
	});
	if (r.error) {
		throw new Error(`plan tool: find failed: ${r.error.message}`);
	}
	const raw = r.stdout ?? "";
	const lines: string[] = raw
		.split("\n")
		.map((line: string) => line.trim())
		.filter((line: string) => line.length > 0);
	const normalized: string[] = lines.map((line: string) => (line.startsWith("./") ? line.slice(2) : line));
	const sorted = [...new Set<string>(normalized)].sort();
	return sorted.slice(0, maxMatches);
}

function normalizeRepoRelativePath(p: string): string {
	return p.trim().replace(/\\/g, "/").replace(/^\.\/+/, "").replace(/\/+/g, "/");
}

function normalizePathAlphaOnly(p: string): string {
	return normalizeRepoRelativePath(p).toLowerCase().replace(/[^a-z]/g, "");
}

/**
 * FIRST-pass resolver requested by user:
 * - Normalize requested path to a-z only
 * - Normalize all planModeReadPaths to a-z only
 * - Compare full normalized path strings
 * Returns real read paths corresponding to matched normalized strings.
 */
function findAlphaOnlyFullPathMatchesInReadPaths(
	planModeReadPaths: string[],
	requestedPath: string,
	maxMatches: number,
): string[] {
	const requestedAlpha = normalizePathAlphaOnly(requestedPath);
	if (!requestedAlpha) return [];
	const normalizedReadPaths = [...new Set(
		planModeReadPaths
			.map((p) => normalizeRepoRelativePath(p))
			.filter((p) => p.length > 0),
	)];
	const matched = normalizedReadPaths
		.filter((p) => normalizePathAlphaOnly(p) === requestedAlpha)
		.sort();
	return matched.slice(0, maxMatches);
}

/**
 * Flexible read-path resolver with deterministic priority:
 * 1) exact normalized path match
 * 2) suffix match by normalized path segments
 * 3) exact filename match (case-insensitive)
 *
 * Returns de-duplicated candidates ordered by score (higher first) and then lexicographically.
 */
function findFlexibleMatchesInReadPaths(
	planModeReadPaths: string[],
	requestedPath: string,
	maxMatches: number,
): string[] {
	const requestedNorm = normalizeRepoRelativePath(requestedPath);
	if (!requestedNorm) return [];
	const requestedBase = basename(requestedNorm).toLowerCase();
	const normalizedReadPaths = [...new Set(
		planModeReadPaths
			.map((p) => normalizeRepoRelativePath(p))
			.filter((p) => p.length > 0),
	)];
	type Scored = { path: string; score: number };
	const scored: Scored[] = [];
	for (const p of normalizedReadPaths) {
		let score = 0;
		const pLower = p.toLowerCase();
		const requestedLower = requestedNorm.toLowerCase();
		if (pLower === requestedLower) score = 300;
		else if (pLower.endsWith("/" + requestedLower) || requestedLower.endsWith("/" + pLower)) score = 200;
		else if (basename(pLower) === requestedBase) score = 100;
		if (score > 0) scored.push({ path: p, score });
	}
	return scored
		.sort((a, b) => (b.score - a.score) || a.path.localeCompare(b.path))
		.map((s) => s.path)
		.slice(0, maxMatches);
}
