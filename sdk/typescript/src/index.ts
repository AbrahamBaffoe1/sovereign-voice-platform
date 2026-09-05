/** Transport-safe HTTP and WebSocket client for Sovereign Voice Platform. */

export type LanguageCode = "tw" | "gaa" | "ee" | "ha" | "en" | (string & {});

export interface LanguageInfo {
  code: string;
  name: string;
  iso639_3?: string | null;
  aliases?: string[];
  asr_ready?: boolean;
  tts_ready?: boolean;
}

export interface TranscriptionResult {
  text: string;
  language?: string | null;
  language_probability?: number | null;
  duration_seconds?: number | null;
}

export interface SpeechOptions {
  language: LanguageCode;
  voiceId?: string;
  pace?: number;
}

export interface VoiceClientOptions {
  baseUrl: string;
  apiKey?: string;
  fetchImpl?: typeof fetch;
}

function trimSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export class VoiceHttpClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch;

  constructor(options: VoiceClientOptions) {
    this.baseUrl = trimSlash(options.baseUrl);
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  private headers(extra?: HeadersInit): Headers {
    const headers = new Headers(extra);
    if (this.apiKey) headers.set("X-Voice-API-Key", this.apiKey);
    return headers;
  }

  private async checked(response: Response): Promise<Response> {
    if (response.ok) return response;
    const body = await response.text();
    throw new Error(`Voice API ${response.status}: ${body || response.statusText}`);
  }

  async listLanguages(): Promise<LanguageInfo[]> {
    const response = await this.checked(
      await this.fetchImpl(`${this.baseUrl}/v1/languages`, { headers: this.headers() }),
    );
    return (await response.json()) as LanguageInfo[];
  }

  async transcribe(audio: Blob, language?: LanguageCode): Promise<TranscriptionResult> {
    const form = new FormData();
    form.append("file", audio, "audio.wav");
    if (language) form.append("language", language);
    const response = await this.checked(
      await this.fetchImpl(`${this.baseUrl}/v1/transcribe`, {
        method: "POST",
        headers: this.headers(),
        body: form,
      }),
    );
    return (await response.json()) as TranscriptionResult;
  }

  async synthesize(text: string, options: SpeechOptions): Promise<ArrayBuffer> {
    const response = await this.checked(
      await this.fetchImpl(`${this.baseUrl}/v1/speech`, {
        method: "POST",
        headers: this.headers({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          text,
          language: options.language,
          voice_id: options.voiceId ?? null,
          pace: options.pace ?? 1,
        }),
      }),
    );
    return response.arrayBuffer();
  }
}

export interface ConversationStart {
  language: LanguageCode;
  sampleRate?: number;
  useLlm?: boolean;
  voiceId?: string;
}

export type ConversationEvent =
  | { type: "ready" }
  | { type: "started" }
  | { type: "interrupted" }
  | { type: "transcript"; text: string; language?: string }
  | { type: "response_text"; text: string }
  | { type: "turn_complete"; timings_ms?: Record<string, number> }
  | { type: "error"; error: string };

export interface ConversationSocketOptions {
  url: string;
  apiKey?: string;
  webSocketFactory?: (url: string) => WebSocket;
  onEvent?: (event: ConversationEvent) => void;
  onAudio?: (chunk: ArrayBuffer) => void;
}

export class VoiceConversationSocket {
  private readonly socket: WebSocket;
  private readonly onEvent: ((event: ConversationEvent) => void) | undefined;
  private readonly onAudio: ((chunk: ArrayBuffer) => void) | undefined;

  constructor(options: ConversationSocketOptions) {
    const url = new URL(options.url);
    if (options.apiKey) url.searchParams.set("api_key", options.apiKey);
    const factory = options.webSocketFactory ?? ((value: string) => new WebSocket(value));
    this.socket = factory(url.toString());
    this.socket.binaryType = "arraybuffer";
    this.onEvent = options.onEvent;
    this.onAudio = options.onAudio;
    this.socket.addEventListener("message", (message: MessageEvent<unknown>) => {
      if (typeof message.data === "string") {
        this.onEvent?.(JSON.parse(message.data) as ConversationEvent);
      } else if (message.data instanceof ArrayBuffer) {
        this.onAudio?.(message.data);
      }
    });
  }

  start(options: ConversationStart): void {
    this.socket.send(
      JSON.stringify({
        type: "start",
        language: options.language,
        sample_rate: options.sampleRate ?? 16000,
        use_llm: options.useLlm ?? true,
        voice_id: options.voiceId ?? null,
      }),
    );
  }

  sendPcm16(chunk: ArrayBuffer | ArrayBufferView): void {
    this.socket.send(chunk);
  }

  commit(): void {
    this.socket.send(JSON.stringify({ type: "commit" }));
  }

  /** Stop pending output and optionally begin a replacement turn immediately. */
  interrupt(next?: ConversationStart): void {
    this.socket.send(JSON.stringify({ type: "interrupt" }));
    if (next) this.start(next);
  }

  close(code = 1000, reason = "client closed"): void {
    this.socket.close(code, reason);
  }
}
