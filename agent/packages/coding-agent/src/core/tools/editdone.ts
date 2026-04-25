import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "@sinclair/typebox";
import type { ToolDefinition } from "../extensions/types.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";

/**
 * `editdone` is a pure signalling tool used in implement mode.
 *
 * The agent loop iterates planned files one-by-one. For each plan the model
 * applies `edit`/`write` calls until it is satisfied, then calls `editdone`.
 * The first `editdone` is a draft completion claim: the loop sends a strict
 * confirmation handshake (original file content, edited content, plan text,
 * and evidence) and asks the model to either:
 *   - call `editdone` again with stronger detailed evidence, or
 *   - continue implementation via `read`/`edit`/`write`.
 * Only a second consecutive detailed `editdone` advances to the next plan.
 * The run ends only after that handshake completes for every planned file.
 *
 * The tool itself performs no filesystem work — it just echoes back a
 * confirmation message. All state transitions happen in the agent loop.
 */
const editdoneSchema = Type.Object(
	{
		filepath: Type.String({
			description: "Path of the planned file whose implementation is being declared complete.",
		}),
		plan: Type.String({
			description: "The plan text for this file (copy verbatim from the plan submission).",
		}),
		completedevidence: Type.String({
			description:
				"Detailed completion evidence: concrete symbols/behaviors changed, why the plan is fully satisfied, and why style matches surrounding code. Used by the per-plan two-step editdone handshake.",
		}),
	},
	{ additionalProperties: false },
);

export type EditDoneToolInput = Static<typeof editdoneSchema>;

export interface EditDoneToolDetails {
	filepath: string;
	completedevidence: string;
}

export function createEditDoneToolDefinition(): ToolDefinition<
	typeof editdoneSchema,
	EditDoneToolDetails,
	Record<string, never>
> {
	return {
		name: "editdone",
		label: "editdone",
		description:
			"Signal completion in IMPLEMENT mode for the CURRENT planned file. This tool uses a two-step handshake: first `editdone` triggers strict confirmation; only a second consecutive detailed `editdone` advances to the next plan. Required payload: { filepath, plan, completedevidence }.",
		promptSnippet:
			"IMPLEMENT handshake signal. Required: { filepath, plan, completedevidence }. First call = draft claim; second consecutive detailed call = advance.",
		promptGuidelines: [
			"Only call `editdone` for the current planned file in IMPLEMENT mode.",
			"`filepath` and `plan` must match the plan the agent is currently asking you to implement.",
			"First `editdone` does not advance; it triggers a strict self-audit handshake in agent-loop.",
			"If the handshake reveals missing work, continue with `read`/`edit`/`write` for the same plan.",
			"Only the second consecutive `editdone` with detailed `completedevidence` advances the plan.",
			"`completedevidence` must be concrete and specific (symbols/logic/behavior/style), not vague.",
		],
		parameters: editdoneSchema,
		async execute(_toolCallId, input: EditDoneToolInput) {
			const filepath = (input.filepath ?? "").trim();
			const completedevidence = (input.completedevidence ?? "").trim();
			return {
				content: [
					{
						type: "text" as const,
						text:
							`editdone received for \`${filepath}\`.\n` +
							`This is an IMPLEMENT handshake signal (state transitions happen in agent-loop).\n` +
							`If this is the first editdone for the current plan, agent-loop will request strict confirmation.\n` +
							`Evidence: ${completedevidence || "(none)"}`,
					},
				],
				details: { filepath, completedevidence },
			};
		},
		renderCall(args, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			const fp = typeof (args as any)?.filepath === "string" ? (args as any).filepath : "";
			text.setText(`${theme.fg("toolTitle", theme.bold("editdone"))} ${theme.fg("accent", fp)}`);
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

export function createEditDoneTool(): AgentTool<typeof editdoneSchema> {
	return wrapToolDefinition(createEditDoneToolDefinition());
}

export const editdoneToolDefinition = createEditDoneToolDefinition();
export const editdoneTool = createEditDoneTool();
