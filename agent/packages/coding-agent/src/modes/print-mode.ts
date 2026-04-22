/**
 * Print mode (single-shot): Send prompts, output result, exit.
 *
 * Used for:
 * - `pi -p "prompt"` - text output
 * - `pi --mode json "prompt"` - JSON event stream
 */

import type { AssistantMessage, ImageContent } from "@mariozechner/pi-ai";
import type { AgentSessionRuntimeHost } from "../core/agent-session-runtime.js";
import { flushRawStdout, writeRawStdout } from "../core/output-guard.js";

/**
 * JSONL line for `--mode json` only. Kept here (not in rpc/jsonl) so print mode stays robust
 * when session events contain cycles, bigint, or symbols — without aborting before `agent_end`.
 */
function stringifyPrintModeJsonLine(value: unknown): string {
	try {
		return JSON.stringify(value, printModeJsonReplacer());
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		return JSON.stringify({ type: "json_serialization_error", error: message });
	}
}

/** json-stringify-safe style: only values on the current ancestor stack are cycles (DAG-safe). */
function printModeJsonReplacer(): (this: unknown, key: string, value: unknown) => unknown {
	const stack: unknown[] = [];
	const keys: string[] = [];

	const cycleReplacer = function (this: unknown, _key: string, value: unknown): string {
		if (stack[0] === value) return "[Circular ~]";
		const idx = stack.indexOf(value);
		return `[Circular ~.${keys.slice(0, idx).join(".")}]`;
	};

	return function (this: unknown, key: string, value: unknown): unknown {
		if (typeof value === "symbol" || typeof value === "function") return undefined;
		if (typeof value === "bigint") return value.toString();

		if (stack.length > 0) {
			const thisPos = stack.indexOf(this);
			if (~thisPos) {
				stack.splice(thisPos + 1);
				keys.splice(thisPos, Number.POSITIVE_INFINITY, key);
			} else {
				stack.push(this);
				keys.push(key);
			}
			if (~stack.indexOf(value)) {
				return cycleReplacer.call(this, key, value);
			}
		} else {
			stack.push(value);
		}

		return value;
	};
}

/**
 * Options for print mode.
 */
export interface PrintModeOptions {
	/** Output mode: "text" for final response only, "json" for all events */
	mode: "text" | "json";
	/** Array of additional prompts to send after initialMessage */
	messages?: string[];
	/** First message to send (may contain @file content) */
	initialMessage?: string;
	/** Images to attach to the initial message */
	initialImages?: ImageContent[];
}

/**
 * Run in print (single-shot) mode.
 * Sends prompts to the agent and outputs the result.
 */
export async function runPrintMode(runtimeHost: AgentSessionRuntimeHost, options: PrintModeOptions): Promise<number> {
	const { mode, messages = [], initialMessage, initialImages } = options;
	let exitCode = 0;
	let session = runtimeHost.session;
	let unsubscribe: (() => void) | undefined;

	const rebindSession = async (): Promise<void> => {
		session = runtimeHost.session;
		await session.bindExtensions({
			commandContextActions: {
				waitForIdle: () => session.agent.waitForIdle(),
				newSession: async (newSessionOptions) => {
					const result = await runtimeHost.newSession(newSessionOptions);
					if (!result.cancelled) {
						await rebindSession();
					}
					return result;
				},
				fork: async (entryId) => {
					const result = await runtimeHost.fork(entryId);
					if (!result.cancelled) {
						await rebindSession();
					}
					return { cancelled: result.cancelled };
				},
				navigateTree: async (targetId, navigateOptions) => {
					const result = await session.navigateTree(targetId, {
						summarize: navigateOptions?.summarize,
						customInstructions: navigateOptions?.customInstructions,
						replaceInstructions: navigateOptions?.replaceInstructions,
						label: navigateOptions?.label,
					});
					return { cancelled: result.cancelled };
				},
				switchSession: async (sessionPath) => {
					const result = await runtimeHost.switchSession(sessionPath);
					if (!result.cancelled) {
						await rebindSession();
					}
					return result;
				},
				reload: async () => {
					await session.reload();
				},
			},
			onError: (err) => {
				console.error(`Extension error (${err.extensionPath}): ${err.error}`);
			},
		});

		unsubscribe?.();
		unsubscribe = session.subscribe((event) => {
			if (mode !== "json") return;
			try {
				writeRawStdout(`${stringifyPrintModeJsonLine(event)}\n`);
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				writeRawStdout(`${JSON.stringify({ type: "print_mode_stdout_error", error: message })}\n`);
			}
		});
	};

	try {
		if (mode === "json") {
			const header = session.sessionManager.getHeader();
			if (header) {
				writeRawStdout(`${stringifyPrintModeJsonLine(header)}\n`);
			}
		}

		await rebindSession();

		if (initialMessage) {
			await session.prompt(initialMessage, { images: initialImages });
		}

		for (const message of messages) {
			await session.prompt(message);
		}

		if (mode === "text") {
			const state = session.state;
			const lastMessage = state.messages[state.messages.length - 1];

			if (lastMessage?.role === "assistant") {
				const assistantMsg = lastMessage as AssistantMessage;
				if (assistantMsg.stopReason === "error" || assistantMsg.stopReason === "aborted") {
					console.error(assistantMsg.errorMessage || `Request ${assistantMsg.stopReason}`);
					exitCode = 1;
				} else {
					for (const content of assistantMsg.content) {
						if (content.type === "text") {
							writeRawStdout(`${content.text}\n`);
						}
					}
				}
			}
		}

		return exitCode;
	} finally {
		unsubscribe?.();
		await runtimeHost.dispose();
		await flushRawStdout();
	}
}
