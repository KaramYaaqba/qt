import { useRef, useState, useCallback } from 'react';
import type { FollowAlongState, PositionUpdate } from '../types';
import { getReciteWebSocketUrl } from '../services/api';

interface Options {
  onWordUpdate: (ayah: number, word: number) => void;
  onEvalReady: (ayah: number, letterResults: NonNullable<PositionUpdate['letter_results']>) => void;
  onDone: () => void;
  onError: (msg: string) => void;
}

export function useFollowAlongRecorder({ onWordUpdate, onEvalReady, onDone, onError }: Options) {
  const [state, setState] = useState<FollowAlongState>('idle');
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const stop = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }

    contextRef.current?.close();
    contextRef.current = null;

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send('END');
    }
  }, []);

  const start = useCallback(async (surahNumber: number) => {
    setState('connecting');

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
    } catch {
      onError('Microphone permission denied');
      setState('idle');
      return;
    }

    streamRef.current = stream;

    const audioContext = new AudioContext({ sampleRate: 16000 });
    contextRef.current = audioContext;

    const ws = new WebSocket(getReciteWebSocketUrl(surahNumber));
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      setState('recording');

      const source = audioContext.createMediaStreamSource(stream);
      // 2048 samples @ 16kHz ≈ 128ms per chunk
      const processor = audioContext.createScriptProcessor(2048, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const float32 = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32767));
        }
        ws.send(int16.buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
    };

    ws.onmessage = (e) => {
      let msg: PositionUpdate;
      try {
        msg = JSON.parse(e.data as string);
      } catch {
        return;
      }

      if (msg.type === 'position' && msg.ayah !== undefined && msg.word !== undefined) {
        onWordUpdate(msg.ayah, msg.word);
        if (msg.completed_ayah && msg.letter_results) {
          onEvalReady(msg.ayah, msg.letter_results);
        }
      } else if (msg.type === 'done') {
        processorRef.current?.disconnect();
        processorRef.current = null;
        stream.getTracks().forEach(t => t.stop());
        audioContext.close();
        setState('done');
        onDone();
      } else if (msg.type === 'error') {
        onError(msg.message ?? 'Unknown error');
        setState('idle');
      }
    };

    ws.onerror = () => {
      onError('Connection error — please try again');
      setState('idle');
    };

    ws.onclose = (e) => {
      if (e.code !== 1000 && state === 'recording') {
        onError('Connection lost — please try again');
        setState('idle');
      }
    };
  }, [onWordUpdate, onEvalReady, onDone, onError, state]);

  return { state, start, stop };
}
