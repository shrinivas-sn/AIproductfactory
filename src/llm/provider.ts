import { z } from "zod";
import crypto from "crypto";
import fs from "fs";
import path from "path";

// ============================================================
// Tier-1.5+ Advanced LLM Infrastructure
// Support for: OpenAI, Anthropic, Gemini, NVIDIA (DeepSeek)
// ============================================================

export type LLMProviderType = "openai" | "anthropic" | "gemini" | "nvidia" | "deepseek";
export type RoutingProfile = "structured" | "reasoning" | "long_context";

export interface LLMMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_call_id?: string;
  name?: string;
}

export interface LLMToolCall {
  id?: string;
  type?: string;
  function?: {
    name: string;
    arguments: string;
  };
  [key: string]: unknown;
}

export interface LLMResponse {
  text: string;
  usage: { promptTokens: number; completionTokens: number };
  model: string;
  provider: LLMProviderType;
  costEstimate: number;
  latency: number; 
  toolCalls?: LLMToolCall[];
}

export interface LLMOptions {
  temperature?: number;
  maxTokens?: number;
  model?: string;
  provider?: LLMProviderType;
  profile?: RoutingProfile;
  timeout?: number;
  useCache?: boolean;
  cacheVersion?: string;
  validateResponse?: (response: LLMResponse) => boolean;
}

export class LLMProviderError extends Error {
  readonly status?: number;
  readonly provider?: LLMProviderType;
  readonly details?: unknown;

  constructor(
    message: string,
    options?: { status?: number; provider?: LLMProviderType; details?: unknown }
  ) {
    super(message);
    this.name = "LLMProviderError";
    this.status = options?.status;
    this.provider = options?.provider;
    this.details = options?.details;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export interface ProviderMetrics {
  success: number;
  failure: number;
  attempts: number;
  successRate: number;
  p50LatencyMs: number | null;
  p90LatencyMs: number | null;
  fallbacks: number;
  healthScore: number;
  lastErrorAt?: number;
}

export interface ProviderSummaryRecord {
  successRate: string;
  p50Latency: string;
  p90Latency: string;
  fallbacks: number;
  healthScore: string;
}

const DEFAULT_REQUEST_TIMEOUT_MS = Number(process.env.LLM_REQUEST_TIMEOUT_MS || 15000);
const DEFAULT_CACHE_TTL_MS = Number(process.env.LLM_CACHE_TTL_MS || 1000 * 60 * 60 * 24);
const DEFAULT_CACHE_VERSION = process.env.LLM_CACHE_VERSION || "v1";
const PROVIDERS: LLMProviderType[] = ["openai", "anthropic", "gemini", "nvidia", "deepseek"];

// ── Routing Profiles ────────────────────────────────────────────────────────

const ROUTING_PROFILES: Record<RoutingProfile, { provider: LLMProviderType; model: string }> = {
  structured: { provider: "openai", model: "gpt-4o-mini" },
  reasoning: { provider: "nvidia", model: "deepseek-ai/deepseek-v4-pro" }, // DeepSeek is excellent for reasoning
  long_context: { provider: "gemini", model: "gemini-2.0-flash" },
};

// ── Telemetry & Observability ────────────────────────────────────────────────

class TelemetryStore {
  private stats: Record<string, { success: number; failure: number; fallbacks: number; latencies: number[]; lastErrorAt?: number }> = {};

  record(provider: string, success: boolean, latency?: number, isFallback = false) {
    if (!this.stats[provider]) {
      this.stats[provider] = { success: 0, failure: 0, fallbacks: 0, latencies: [] };
    }
    if (success) {
      this.stats[provider].success++;
      if (latency) this.stats[provider].latencies.push(latency);
    } else {
      this.stats[provider].failure++;
      this.stats[provider].lastErrorAt = Date.now();
    }
    if (isFallback) this.stats[provider].fallbacks++;
  }

  getMetrics(): Record<string, ProviderMetrics> {
    const summary: Record<string, ProviderMetrics> = {};
    for (const [p, s] of Object.entries(this.stats)) {
      const sortedLatencies = [...s.latencies].sort((a, b) => a - b);
      const attempts = s.success + s.failure;
      summary[p] = {
        success: s.success,
        failure: s.failure,
        attempts,
        successRate: attempts ? s.success / attempts : 1,
        p50LatencyMs: percentile(sortedLatencies, 0.5),
        p90LatencyMs: percentile(sortedLatencies, 0.9),
        fallbacks: s.fallbacks,
        healthScore: this.getHealthScore(p),
        lastErrorAt: s.lastErrorAt,
      };
    }
    return summary;
  }

  getSummary(): Record<string, ProviderSummaryRecord> {
    const metrics = this.getMetrics();
    return Object.fromEntries(Object.entries(metrics).map(([provider, m]) => [
      provider,
      {
        successRate: `${(m.successRate * 100).toFixed(1)}%`,
        p50Latency: m.p50LatencyMs === null ? "N/A" : `${m.p50LatencyMs}ms`,
        p90Latency: m.p90LatencyMs === null ? "N/A" : `${m.p90LatencyMs}ms`,
        fallbacks: m.fallbacks,
        healthScore: m.healthScore.toFixed(3),
      },
    ]));
  }

  getHealthScore(provider: string): number {
    const s = this.stats[provider];
    if (!s) return 1;
    const attempts = s.success + s.failure;
    const successRate = attempts ? s.success / attempts : 1;
    const p90 = percentile([...s.latencies].sort((a, b) => a - b), 0.9) ?? DEFAULT_REQUEST_TIMEOUT_MS;
    const latencyPenalty = Math.min(p90 / DEFAULT_REQUEST_TIMEOUT_MS, 2) * 0.25;
    const fallbackPenalty = attempts ? Math.min(s.fallbacks / attempts, 1) * 0.1 : 0;
    return Math.max(0.05, successRate - latencyPenalty - fallbackPenalty);
  }

  toPrometheus(prefix = "llm"): string {
    const lines: string[] = [];
    for (const [provider, m] of Object.entries(this.getMetrics())) {
      const label = `{provider="${provider}"}`;
      lines.push(`${prefix}_requests_total${label} ${m.attempts}`);
      lines.push(`${prefix}_success_total${label} ${m.success}`);
      lines.push(`${prefix}_failure_total${label} ${m.failure}`);
      lines.push(`${prefix}_fallback_total${label} ${m.fallbacks}`);
      lines.push(`${prefix}_success_rate${label} ${m.successRate}`);
      lines.push(`${prefix}_health_score${label} ${m.healthScore}`);
      if (m.p50LatencyMs !== null) lines.push(`${prefix}_latency_p50_ms${label} ${m.p50LatencyMs}`);
      if (m.p90LatencyMs !== null) lines.push(`${prefix}_latency_p90_ms${label} ${m.p90LatencyMs}`);
    }
    return lines.join("\n");
  }
}

export const telemetry = new TelemetryStore();

function percentile(sorted: number[], p: number): number | null {
  if (!sorted.length) return null;
  const index = Math.min(sorted.length - 1, Math.floor(sorted.length * p));
  return sorted[index];
}

// ── Concurrency Control (Semaphore) ──────────────────────────────────────────

class Semaphore {
  private active = 0;
  private queue: (() => void)[] = [];
  constructor(private limit: number) {}

  async acquire() {
    if (this.active < this.limit) {
      this.active++;
      return;
    }
    return new Promise<void>(resolve => this.queue.push(resolve));
  }

  release() {
    this.active--;
    if (this.queue.length > 0) {
      this.active++;
      const next = this.queue.shift();
      if (next) next();
    }
  }
}

const concurrencyLimiters: Record<string, Semaphore> = {
  openai: new Semaphore(5),
  anthropic: new Semaphore(5),
  gemini: new Semaphore(5),
  nvidia: new Semaphore(3), // DeepSeek is powerful, but keep concurrency measured
};

// ── Persistent Cache Layer ────────────────────────────────────────────────────

class FileCache {
  private cachePath = path.join(process.cwd(), ".cache", "llm_cache.json");
  private data: Record<string, { response: LLMResponse; timestamp: number }> = {};

  constructor() { this.load(); }

  private load() {
    try {
      if (!fs.existsSync(path.dirname(this.cachePath))) fs.mkdirSync(path.dirname(this.cachePath), { recursive: true });
      if (fs.existsSync(this.cachePath)) this.data = JSON.parse(fs.readFileSync(this.cachePath, "utf-8"));
    } catch (e) { console.warn("[Cache] Failed to load persistent cache:", e); }
  }

  private save() {
    try {
      const tmpPath = `${this.cachePath}.${process.pid}.tmp`;
      fs.writeFileSync(tmpPath, JSON.stringify(this.data, null, 2));
      fs.renameSync(tmpPath, this.cachePath);
    }
    catch (e) { console.warn("[Cache] Failed to save persistent cache:", e); }
  }

  generateKey(messages: LLMMessage[], options?: LLMOptions): string {
    const data = JSON.stringify({
      version: options?.cacheVersion || DEFAULT_CACHE_VERSION,
      messages,
      options: {
        model: options?.model,
        provider: options?.provider,
        profile: options?.profile,
        temp: options?.temperature,
        maxTokens: options?.maxTokens,
      },
    });
    return crypto.createHash("sha256").update(data).digest("hex");
  }

  get(key: string): LLMResponse | null {
    const entry = this.data[key];
    if (entry && Date.now() - entry.timestamp < DEFAULT_CACHE_TTL_MS) return entry.response;
    return null;
  }

  set(key: string, response: LLMResponse) {
    this.data[key] = { response, timestamp: Date.now() };
    this.save();
  }
}

export const llmCache = new FileCache();

// ── Adapters ─────────────────────────────────────────────────────────────────

const PRICING: Record<string, { input: number; output: number }> = {
  "gpt-4o": { input: 2.50, output: 10.00 },
  "gpt-4o-mini": { input: 0.15, output: 0.60 },
  "claude-3-7-sonnet-latest": { input: 3.00, output: 15.00 },
  "claude-3-5-haiku-latest": { input: 0.25, output: 1.25 },
  "gemini-2.0-flash": { input: 0.10, output: 0.40 },
  "deepseek-ai/deepseek-v4-pro": { input: 0.50, output: 2.00 }, // Estimated NVIDIA pricing
};

function estimateCost(model: string, promptTokens: number, completionTokens: number): number {
  const price = PRICING[model] || PRICING["gpt-4o-mini"];
  return (promptTokens / 1_000_000) * price.input + (completionTokens / 1_000_000) * price.output;
}

abstract class LLMAdapter {
  abstract generate(messages: LLMMessage[], options?: LLMOptions): Promise<LLMResponse>;
}

class OpenAIAdapter extends LLMAdapter {
  async generate(messages: LLMMessage[], options?: LLMOptions): Promise<LLMResponse> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new LLMProviderError("OPENAI_API_KEY is not set", { provider: "openai" });
    const model = options?.model || "gpt-4o-mini";
    const start = Date.now();

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model,
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        temperature: options?.temperature ?? 0.7,
      }),
      signal: options?.timeout ? AbortSignal.timeout(options.timeout) : undefined,
    });

    if (!response.ok) {
      throw new LLMProviderError(`OpenAI: ${response.statusText}`, { status: response.status, provider: "openai" });
    }
    const data = await response.json();
    const message = data.choices?.[0]?.message;
    const text = message?.content || message?.reasoning_content || message?.reasoning || "";
    return {
      text,
      usage: { promptTokens: data.usage?.prompt_tokens ?? 0, completionTokens: data.usage?.completion_tokens ?? 0 },
      model: data.model,
      provider: "openai",
      costEstimate: estimateCost(data.model, data.usage?.prompt_tokens ?? 0, data.usage?.completion_tokens ?? 0),
      latency: Date.now() - start,
    };
  }
}

class NvidiaAdapter extends LLMAdapter {
  async generate(messages: LLMMessage[], options?: LLMOptions): Promise<LLMResponse> {
    const apiKey = process.env.NVIDIA_API_KEY;
    if (!apiKey) throw new LLMProviderError("NVIDIA_API_KEY is not set", { provider: "nvidia" });
    const model = options?.model || "deepseek-ai/deepseek-v4-pro";
    const baseUrl = process.env.NVIDIA_BASE_URL || "https://integrate.api.nvidia.com/v1";
    const start = Date.now();

    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model,
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        temperature: options?.temperature ?? 1,
        max_tokens: options?.maxTokens ?? 16384,
        extra_body: { chat_template_kwargs: { thinking: false } }, // As requested in snippet
      }),
      signal: options?.timeout ? AbortSignal.timeout(options.timeout) : undefined,
    });

    if (!response.ok) {
      const err = (await response.json().catch(() => ({}))) as { message?: string };
      throw new LLMProviderError(`NVIDIA: ${err.message || response.statusText}`, { status: response.status, provider: "nvidia", details: err });
    }
    const data = await response.json();
    const promptTokens = data.usage?.prompt_tokens || 0;
    const completionTokens = data.usage?.completion_tokens || 0;
    const message = data.choices?.[0]?.message;
    const text = message?.content || message?.reasoning_content || message?.reasoning || "";

    return {
      text,
      usage: { promptTokens, completionTokens },
      model,
      provider: "nvidia",
      costEstimate: estimateCost(model, promptTokens, completionTokens),
      latency: Date.now() - start,
    };
  }
}

// ... Anthropic and Gemini adapters remain same ...
class AnthropicAdapter extends LLMAdapter {
  async generate(messages: LLMMessage[], options?: LLMOptions): Promise<LLMResponse> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new LLMProviderError("ANTHROPIC_API_KEY is not set", { provider: "anthropic" });
    const model = options?.model || "claude-3-5-haiku-latest";
    const start = Date.now();

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": apiKey, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({
        model,
        system: messages.find(m => m.role === "system")?.content,
        messages: messages.filter(m => m.role !== "system").map(m => ({ role: m.role === "assistant" ? "assistant" : "user", content: m.content })),
        max_tokens: 2048,
      }),
      signal: options?.timeout ? AbortSignal.timeout(options.timeout) : undefined,
    });

    if (!response.ok) {
      throw new LLMProviderError(`Anthropic: ${response.statusText}`, { status: response.status, provider: "anthropic" });
    }
    const data = await response.json();
    return {
      text: data.content[0].text,
      usage: { promptTokens: data.usage.input_tokens, completionTokens: data.usage.output_tokens },
      model: data.model,
      provider: "anthropic",
      costEstimate: estimateCost(data.model, data.usage.input_tokens, data.usage.output_tokens),
      latency: Date.now() - start,
    };
  }
}

class GeminiAdapter extends LLMAdapter {
  async generate(messages: LLMMessage[], options?: LLMOptions): Promise<LLMResponse> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new LLMProviderError("GEMINI_API_KEY is not set", { provider: "gemini" });
    const model = options?.model || "gemini-2.0-flash";
    const start = Date.now();

    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: messages.filter(m => m.role !== "system").map(m => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] })),
        system_instruction: { parts: [{ text: messages.find(m => m.role === "system")?.content || "" }] },
      }),
      signal: options?.timeout ? AbortSignal.timeout(options.timeout) : undefined,
    });

    if (!response.ok) {
      throw new LLMProviderError(`Gemini: ${response.statusText}`, { status: response.status, provider: "gemini" });
    }
    const data = await response.json();
    return {
      text: data.candidates[0].content.parts[0].text,
      usage: { promptTokens: data.usageMetadata.promptTokenCount, completionTokens: data.usageMetadata.candidatesTokenCount },
      model,
      provider: "gemini",
      costEstimate: estimateCost(model, data.usageMetadata.promptTokenCount, data.usageMetadata.candidatesTokenCount),
      latency: Date.now() - start,
    };
  }
}

class DeepSeekAdapter extends LLMAdapter {
  async generate(messages: LLMMessage[], options?: LLMOptions): Promise<LLMResponse> {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey) throw new LLMProviderError("DEEPSEEK_API_KEY is not set", { provider: "deepseek" });
    const model = options?.model || "deepseek-chat";
    const baseUrl = process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com";
    const start = Date.now();

    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model,
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        temperature: options?.temperature ?? 0.7,
        max_tokens: options?.maxTokens ?? 4096,
      }),
      signal: options?.timeout ? AbortSignal.timeout(options.timeout) : undefined,
    });

    if (!response.ok) {
      const err = (await response.json().catch(() => ({}))) as { message?: string };
      throw new LLMProviderError(`DeepSeek: ${err.message || response.statusText}`, { status: response.status, provider: "deepseek", details: err });
    }
    const data = await response.json();
    const promptTokens = data.usage?.prompt_tokens || 0;
    const completionTokens = data.usage?.completion_tokens || 0;
    const message = data.choices?.[0]?.message;
    const text = message?.content || message?.reasoning_content || message?.reasoning || "";

    return {
      text,
      usage: { promptTokens, completionTokens },
      model,
      provider: "deepseek",
      costEstimate: estimateCost(model, promptTokens, completionTokens),
      latency: Date.now() - start,
    };
  }
}

export class LLMManager {
  private adapters: Record<LLMProviderType, LLMAdapter> = {
    openai: new OpenAIAdapter(),
    anthropic: new AnthropicAdapter(),
    gemini: new GeminiAdapter(),
    nvidia: new NvidiaAdapter(),
    deepseek: new DeepSeekAdapter(),
  };

  async generate(messages: LLMMessage[], options?: LLMOptions): Promise<LLMResponse> {
    const effectiveOptions = { ...options, timeout: options?.timeout ?? DEFAULT_REQUEST_TIMEOUT_MS };
    const cacheKey = effectiveOptions.useCache !== false ? llmCache.generateKey(messages, effectiveOptions) : null;
    if (cacheKey) {
      const cached = llmCache.get(cacheKey);
      if (cached && (!effectiveOptions.validateResponse || effectiveOptions.validateResponse(cached))) return cached;
    }

    let profile = effectiveOptions.profile;
    if (!profile) {
      const isComplex = messages.some(m => m.content.length > 5000);
      profile = isComplex ? "long_context" : "structured";
    }
    const route = ROUTING_PROFILES[profile];
    
    const configuredProvider = process.env.DEFAULT_LLM_PROVIDER as LLMProviderType | undefined;
    const primaryProvider = effectiveOptions.provider || configuredProvider || route.provider;
    const model = effectiveOptions.model || process.env.DEFAULT_LLM_MODEL || (primaryProvider === route.provider ? route.model : undefined);
    
    const fallbackProviders = PROVIDERS
      .filter(p => p !== primaryProvider)
      .sort((a, b) => telemetry.getHealthScore(b) - telemetry.getHealthScore(a));
    const providers: LLMProviderType[] = [primaryProvider, ...fallbackProviders];

    let lastError: unknown;
    for (const p of providers) {
      const adapter = this.adapters[p];
      const limiter = concurrencyLimiters[p] || new Semaphore(5);
      
      await limiter.acquire();
      try {
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            const resp = await adapter.generate(messages, { ...effectiveOptions, model: p === primaryProvider ? model : undefined });
            if (effectiveOptions.validateResponse && !effectiveOptions.validateResponse(resp)) {
              throw new LLMProviderError("LLM response failed validation", { status: 422, provider: p });
            }
            if (cacheKey) llmCache.set(cacheKey, resp);
            telemetry.record(p, true, resp.latency, p !== primaryProvider);
            return resp;
          } catch (e: unknown) {
            lastError = e;
            const status = e instanceof LLMProviderError ? e.status : (typeof e === "object" && e !== null && "status" in e ? Number((e as { status?: unknown }).status) : undefined);
            if (status === 401 || status === 400) throw e;
            if (attempt < 2) await new Promise(r => setTimeout(r, Math.pow(2, attempt) * 1000));
          }
        }
        telemetry.record(p, false, undefined, p !== primaryProvider);
      } finally { limiter.release(); }
    }
    if (lastError instanceof Error) {
      throw lastError;
    }
    throw new LLMProviderError(String(lastError || "All LLM providers failed to respond"));
  }

  async generateJSON<T>(schema: z.ZodType<T>, prompt: string, systemPrompt?: string, options?: LLMOptions): Promise<T> {
    const messages: LLMMessage[] = [];
    if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
    messages.push({ role: "user", content: `${prompt}\n\nReturn ONLY valid JSON.` });

    try {
      const validateResponse = (resp: LLMResponse) => {
        const text = resp.text.replace(/```json\n?|```/g, "").trim();
        try {
          return schema.safeParse(JSON.parse(text)).success;
        } catch {
          return false;
        }
      };
      const resp = await this.generate(messages, { profile: "structured", ...options, validateResponse });
      const text = resp.text.replace(/```json\n?|```/g, "").trim();
      const parsed = schema.parse(JSON.parse(text));
      console.log(`[LLM] Result: $${resp.costEstimate.toFixed(6)} | ${resp.latency}ms | ${resp.model}`);
      return parsed;
    } catch (e) {
      console.error("[LLM] generateJSON failed:", e);
      throw e;
    }
  }

  getTelemetry(): Record<string, ProviderMetrics> { return telemetry.getMetrics(); }
  exportPrometheusMetrics(): string { return telemetry.toPrometheus(); }
  printTelemetry(): void { console.table(telemetry.getSummary()); }
}

export const llm = new LLMManager();
export const generate = async (p: string, s?: string, o?: LLMOptions) => (await llm.generate([{ role: "system", content: s || "" }, { role: "user", content: p }], o)).text;

export async function generateJSON<T = unknown>(prompt: string, systemPrompt?: string, options?: LLMOptions): Promise<T> {
  return llm.generateJSON(z.any() as z.ZodType<T>, prompt, systemPrompt, options);
}

export async function embed(text: string): Promise<number[]> {
  return textToVector(text);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
}

function textToVector(text: string): number[] {
  const dim = 128;
  const vector = new Array(dim).fill(0);
  const normalized = text.toLowerCase().replace(/[^a-z0-9\s]/g, "");

  for (let i = 0; i < normalized.length; i++) {
    const idx = i % dim;
    vector[idx] += normalized.charCodeAt(i) / 127;
  }

  const mag = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0)) || 1;
  return vector.map((value) => value / mag);
}
