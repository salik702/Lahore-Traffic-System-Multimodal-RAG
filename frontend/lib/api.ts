export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";
export const MEDIA_URL = API_URL.replace(/\/api\/v1\/?$/, "");

const WINDOWS_PATH = /^[a-zA-Z]:[\\/]/;

export function resolveMediaUrl(value?: string | null): string | null {
  const candidate = value?.trim();
  if (!candidate || WINDOWS_PATH.test(candidate) || candidate.startsWith("file:")) return null;
  if (candidate.split(/[\\/]/).includes("..")) return null;
  try {
    const absolute = new URL(candidate);
    return absolute.protocol === "http:" || absolute.protocol === "https:" ? absolute.toString() : null;
  } catch {
    const path = candidate.includes("/") ? candidate.replaceAll("\\", "/") : `/media/signs/${encodeURIComponent(candidate)}`;
    return new URL(path.startsWith("/") ? path : `/${path}`, `${MEDIA_URL}/`).toString();
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export type Citation = {
  id: string; title: string; organization: string; url: string; page?: number;
  section?: string; region: string; official: boolean; modality: string; excerpt?: string; score?: number;
};
export type Evidence = { id: string; title: string; text: string; modality: string; score: number; source_id: string; metadata: Record<string, unknown> };
export type ChatResponse = {
  answer: string; intent: string; citations: Citation[]; evidence: Evidence[];
  trace: { label: string; detail: string; status: string }[]; retrieval_engine: string; model?: string; session_id: string;
  images: { url: string; title: string; description: string; category: string; source: string; source_url: string; region: string; license?: string; license_url?: string; author?: string; date?: string; date_kind?: string; location?: string; record_id?: string; verified_record_match?: boolean; width?: number; height?: number }[];
  query_understanding: Record<string, unknown>;
};
