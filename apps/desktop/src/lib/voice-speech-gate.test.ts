import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { monitorSpeechDuringPlayback } from './voice-barge-in'
import { downsampleTo16k, floatToPcm16Base64, startSpeechGate } from './voice-speech-gate'

// Same fake-mic harness as voice-barge-in.test.ts: `micLevel` is the analyser's
// byte-domain RMS level and each advance(ms) runs one frame per 16ms.
let now = 0
let micLevel = 0
let frames: FrameRequestCallback[] = []

beforeEach(() => {
  now = 1_000_000
  micLevel = 0
  frames = []
  vi.spyOn(Date, 'now').mockImplementation(() => now)
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => frames.push(cb))
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
  vi.stubGlobal(
    'AudioContext',
    class {
      createAnalyser() {
        return {
          fftSize: 256,
          getByteTimeDomainData: (data: Uint8Array) => data.fill(128 + Math.round(micLevel * 42))
        }
      }

      createMediaStreamSource() {
        return { connect: () => undefined }
      }

      close() {
        return Promise.resolve()
      }
    }
  )
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getTracks: () => [{ stop: () => undefined }] }) }
  })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

async function flushMicrotasks() {
  for (let i = 0; i < 5; i += 1) {
    await Promise.resolve()
  }
}

function advance(ms: number) {
  for (let elapsed = 0; elapsed < ms; elapsed += 16) {
    now += 16
    const pending = frames
    frames = []
    pending.forEach(cb => cb(now))
  }
}

/** A loud sound over playback; `isSpeech` is what the gateway verdict says about it. */
async function loudSound(opts: { speechGate: boolean; isSpeech: () => boolean }) {
  let playing = false
  const onSpeech = vi.fn()
  const stopGate = vi.fn()

  const stop = monitorSpeechDuringPlayback({
    isPlaying: () => playing,
    onSpeech,
    speechGate: opts.speechGate,
    createSpeechGate: () => ({ isSpeech: opts.isSpeech, probability: () => null, stop: stopGate })
  })

  await flushMicrotasks()
  advance(600)
  playing = true
  advance(700)
  micLevel = 0.6 // well above any trigger level
  advance(800)
  stop()

  return { onSpeech, stopGate }
}

describe('barge-in speech gate', () => {
  it('ignores a loud sound the gateway scores as not speech (clatter, running water)', async () => {
    const { onSpeech } = await loudSound({ speechGate: true, isSpeech: () => false })

    expect(onSpeech).not.toHaveBeenCalled()
  })

  it('still interrupts when the loud sound is scored as speech', async () => {
    const { onSpeech } = await loudSound({ speechGate: true, isSpeech: () => true })

    expect(onSpeech).toHaveBeenCalledTimes(1)
  })

  it('is loudness-only when the gate is off, whatever a gate would say', async () => {
    const { onSpeech } = await loudSound({ speechGate: false, isSpeech: () => false })

    expect(onSpeech).toHaveBeenCalledTimes(1)
  })

  it('recovers once the verdict flips to speech mid-sound', async () => {
    let speech = false
    let playing = false
    const onSpeech = vi.fn()

    const stop = monitorSpeechDuringPlayback({
      isPlaying: () => playing,
      onSpeech,
      speechGate: true,
      createSpeechGate: () => ({ isSpeech: () => speech, probability: () => null, stop: () => undefined })
    })

    await flushMicrotasks()
    advance(600)
    playing = true
    advance(700)
    micLevel = 0.6
    advance(600) // loud, scored as noise: no trip
    expect(onSpeech).not.toHaveBeenCalled()
    speech = true
    advance(600)
    stop()

    expect(onSpeech).toHaveBeenCalledTimes(1)
  })

  it('stops the gate tap on cleanup', async () => {
    const { stopGate } = await loudSound({ speechGate: true, isSpeech: () => false })

    expect(stopGate).toHaveBeenCalled()
  })
})

// A minimal audio graph: the gate connects source -> processor -> mute -> destination
// and the test fires `onaudioprocess` itself.
function fakeAudio(sampleRate = 48_000) {
  const processor: { onaudioprocess: ((e: unknown) => void) | null; connect: () => void; disconnect: () => void } = {
    onaudioprocess: null,
    connect: () => undefined,
    disconnect: () => undefined
  }

  const context = {
    sampleRate,
    destination: {},
    createScriptProcessor: () => processor,
    createGain: () => ({ gain: { value: 1 }, connect: () => undefined, disconnect: () => undefined })
  } as unknown as AudioContext

  const source = { connect: () => undefined, disconnect: () => undefined } as unknown as MediaStreamAudioSourceNode

  const push = (samples: number) =>
    processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array(samples).fill(0.1) } })

  return { context, source, push }
}

describe('startSpeechGate (client)', () => {
  it('fails open before any verdict and ships 16 kHz PCM with a reset on the first call', async () => {
    const { context, source, push } = fakeAudio()
    const request = vi.fn().mockResolvedValue({ available: true, probability: 0.9, speech: true })
    const gate = startSpeechGate({ context, source, request, now: () => now, threshold: 0.6 })

    expect(gate.isSpeech()).toBe(true)
    expect(gate.probability()).toBeNull()

    now += 200
    push(4800) // 100 ms @ 48 kHz -> 1600 samples @ 16 kHz
    await flushMicrotasks()

    expect(request).toHaveBeenCalledTimes(1)
    const [method, params] = request.mock.calls[0] as [string, Record<string, unknown>]

    expect(method).toBe('voice.speech_prob')
    expect(params).toMatchObject({ sample_rate: 16_000, threshold: 0.6, reset: true })
    // 1600 int16 samples = 3200 bytes = 4268 base64 chars (with padding)
    expect((params.pcm as string).length).toBe(Math.ceil((1600 * 2) / 3) * 4)
    expect(gate.probability()).toBe(0.9)
  })

  it('reports not-speech when the gateway says so, and goes stale back to open', async () => {
    const { context, source, push } = fakeAudio(16_000)
    const request = vi.fn().mockResolvedValue({ available: true, probability: 0.02, speech: false })
    const gate = startSpeechGate({ context, source, request, now: () => now })

    now += 200
    push(1600)
    await flushMicrotasks()
    expect(gate.isSpeech()).toBe(false)

    now += 5_000 // link stalls: a stale veto must not stick
    expect(gate.isSpeech()).toBe(true)
  })

  it('stays open for good when the backend cannot score (VAD unavailable)', async () => {
    const { context, source, push } = fakeAudio(16_000)
    const request = vi.fn().mockResolvedValue({ available: false })
    const gate = startSpeechGate({ context, source, request, now: () => now })

    now += 200
    push(1600)
    await flushMicrotasks()
    now += 200
    push(1600)
    await flushMicrotasks()

    expect(request).toHaveBeenCalledTimes(1) // gave up, no more calls
    expect(gate.isSpeech()).toBe(true)
  })

  it('gives up after repeated request failures instead of vetoing', async () => {
    const { context, source, push } = fakeAudio(16_000)
    const request = vi.fn().mockRejectedValue(new Error('old backend: unknown method'))
    const gate = startSpeechGate({ context, source, request, now: () => now })

    for (let i = 0; i < 8; i += 1) {
      now += 200
      push(1600)
      await flushMicrotasks()
    }

    expect(request.mock.calls.length).toBe(5)
    expect(gate.isSpeech()).toBe(true)
  })

  it('does not send after stop()', async () => {
    const { context, source, push } = fakeAudio(16_000)
    const request = vi.fn().mockResolvedValue({ available: true, probability: 0.9, speech: true })
    const gate = startSpeechGate({ context, source, request, now: () => now })

    gate.stop()
    now += 200
    push(1600)
    await flushMicrotasks()

    expect(request).not.toHaveBeenCalled()
  })
})

describe('pcm helpers', () => {
  it('downsample keeps 16 kHz input untouched and averages higher rates', () => {
    const same = new Float32Array([0.5, -0.5])

    expect(downsampleTo16k(same, 16_000)).toBe(same)
    const out = downsampleTo16k(new Float32Array([1, 1, 1, 0, 0, 0]), 48_000)

    expect(Array.from(out)).toEqual([1, 0])
  })

  it('encodes int16 little-endian base64', () => {
    // 0.5 -> 0x3fff (LE: ff 3f), -1 -> -32768 (LE: 00 80)
    expect(floatToPcm16Base64(new Float32Array([0.5, -1]))).toBe(btoa(String.fromCharCode(0xff, 0x3f, 0x00, 0x80)))
  })
})
