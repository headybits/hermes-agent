// Client side of the barge-in speech gate. Loudness alone trips barge-in on pan
// clanks and running water; this taps the monitor's mic stream, ships 16 kHz PCM
// to the gateway (`voice.speech_prob`, bundled Silero VAD) and keeps the latest
// "is this human speech" verdict for the trigger to consult.
//
// Fail-open by design: until the first verdict arrives, or if the gateway can't
// score (older backend, VAD unavailable, request errors), `isSpeech()` answers
// true so barge-in degrades to the old loudness-only behavior, never to deaf.

import { $gateway } from '@/store/gateway'

export const SPEECH_GATE_RATE = 16_000
const SEND_INTERVAL_MS = 100
// A verdict older than this is treated as unknown (fail-open) so a stalled link
// can't leave a stale "not speech" veto in place.
const VERDICT_MAX_AGE_MS = 600
// Consecutive request failures before the gate gives up and stays open.
const MAX_FAILURES = 5

export interface SpeechGate {
  /** Latest verdict; true when unknown or stale (fail-open). */
  isSpeech: () => boolean
  /** Latest speech probability, or null when unknown. */
  probability: () => null | number
  stop: () => void
}

export type SpeechProbRequester = <T>(method: string, params: Record<string, unknown>) => Promise<T>

interface SpeechProbResult {
  available: boolean
  probability?: null | number
  speech?: boolean | null
}

const defaultRequester: SpeechProbRequester = async <T>(method: string, params: Record<string, unknown>) => {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Hermes gateway unavailable')
  }

  return gateway.request<T>(method, params)
}

/** Average-pool downsample to 16 kHz mono (same approach as the wake capture). */
export function downsampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === SPEECH_GATE_RATE) {
    return input
  }

  if (inputRate <= 0) {
    return new Float32Array(0)
  }

  const ratio = inputRate / SPEECH_GATE_RATE
  const outLen = Math.max(1, Math.floor(input.length / ratio))
  const out = new Float32Array(outLen)

  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio)
    const end = Math.min(input.length, Math.floor((i + 1) * ratio))
    let sum = 0
    let count = 0

    for (let j = start; j < end; j++) {
      sum += input[j] ?? 0
      count++
    }

    out[i] = count > 0 ? sum / count : 0
  }

  return out
}

export function floatToPcm16Base64(input: Float32Array): string {
  const bytes = new Uint8Array(input.length * 2)
  const view = new DataView(bytes.buffer)

  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i] ?? 0))
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }

  let binary = ''
  const chunk = 0x8000

  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }

  return btoa(binary)
}

export interface StartSpeechGateOptions {
  context: AudioContext
  source: MediaStreamAudioSourceNode
  /** Silero probability at or above which audio counts as speech. */
  threshold?: number
  request?: SpeechProbRequester
  now?: () => number
}

/**
 * Tap `source` and keep a rolling speech verdict from the gateway. The caller owns
 * the AudioContext; `stop()` only disconnects this tap.
 */
export function startSpeechGate(options: StartSpeechGateOptions): SpeechGate {
  const { context, source } = options
  const request = options.request ?? defaultRequester
  const now = options.now ?? Date.now
  const threshold = options.threshold ?? 0.5

  let stopped = false
  let verdict: { at: number; probability: number; speech: boolean } | null = null
  let failures = 0
  let gaveUp = false
  let inFlight = false
  let pending: Float32Array[] = []
  let pendingSamples = 0
  let first = true
  let lastSentAt = 0

  // ScriptProcessor is deprecated but is what the wake capture already uses in
  // this app, and it keeps this tap a few lines instead of an AudioWorklet file.
  const processor = context.createScriptProcessor(2048, 1, 1)
  const mute = context.createGain()

  mute.gain.value = 0

  const flush = () => {
    if (stopped || gaveUp || inFlight || pendingSamples === 0) {
      return
    }

    const merged = new Float32Array(pendingSamples)
    let offset = 0

    for (const part of pending) {
      merged.set(part, offset)
      offset += part.length
    }

    pending = []
    pendingSamples = 0
    inFlight = true
    lastSentAt = now()

    const reset = first

    first = false

    request<SpeechProbResult>('voice.speech_prob', {
      pcm: floatToPcm16Base64(merged),
      sample_rate: SPEECH_GATE_RATE,
      threshold,
      ...(reset ? { reset: true } : {})
    })
      .then(result => {
        failures = 0

        if (!result.available || typeof result.probability !== 'number') {
          // Backend can't score: stay open (loudness-only) for good.
          gaveUp = true
          verdict = null

          return
        }

        verdict = { at: now(), probability: result.probability, speech: result.speech === true }
      })
      .catch(() => {
        failures += 1

        if (failures >= MAX_FAILURES) {
          gaveUp = true
          verdict = null
        }
      })
      .finally(() => {
        inFlight = false
      })
  }

  processor.onaudioprocess = event => {
    if (stopped) {
      return
    }

    const channel = event.inputBuffer.getChannelData(0)
    const chunk = downsampleTo16k(new Float32Array(channel), context.sampleRate)

    pending.push(chunk)
    pendingSamples += chunk.length

    if (now() - lastSentAt >= SEND_INTERVAL_MS) {
      flush()
    }
  }

  source.connect(processor)
  processor.connect(mute)
  mute.connect(context.destination)

  return {
    isSpeech: () => {
      if (gaveUp || !verdict || now() - verdict.at > VERDICT_MAX_AGE_MS) {
        return true
      }

      return verdict.speech
    },
    probability: () => (verdict && now() - verdict.at <= VERDICT_MAX_AGE_MS ? verdict.probability : null),
    stop: () => {
      stopped = true
      processor.onaudioprocess = null

      try {
        source.disconnect(processor)
        processor.disconnect()
        mute.disconnect()
      } catch {
        // already disconnected
      }
    }
  }
}
