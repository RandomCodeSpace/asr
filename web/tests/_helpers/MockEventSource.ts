type Listener = (ev: MessageEvent) => void;

export class MockEventSource {
  url: string;
  readyState: number = 0;  // CONNECTING
  onopen: ((ev: Event) => void) | null = null;
  onmessage: Listener | null = null;
  onerror: ((ev: Event) => void) | null = null;
  private listeners: Map<string, Listener[]> = new Map();
  private static instances: MockEventSource[] = [];

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
    setTimeout(() => {
      this.readyState = 1;  // OPEN
      this.onopen?.(new Event('open'));
    }, 0);
  }

  addEventListener(type: string, listener: Listener) {
    const list = this.listeners.get(type) ?? [];
    list.push(listener);
    this.listeners.set(type, list);
  }

  emit(data: string) {
    const ev = new MessageEvent('message', { data });
    this.onmessage?.(ev);
    this.listeners.get('message')?.forEach((l) => l(ev));
  }

  close() {
    this.readyState = 2;
  }

  static lastInstance(): MockEventSource | undefined {
    return MockEventSource.instances[MockEventSource.instances.length - 1];
  }

  static reset() {
    MockEventSource.instances = [];
  }
}
