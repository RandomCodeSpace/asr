export default function App() {
  return (
    <div style={{ padding: 'var(--s-6)' }}>
      <h1 style={{
        fontSize: 'var(--t-display)',
        fontWeight: 500,
        letterSpacing: '-0.018em',
        color: 'var(--ink-1)',
        marginBottom: 'var(--s-3)',
      }}>
        ASR Operator Console
      </h1>
      <p style={{
        fontSize: 'var(--t-body)',
        color: 'var(--ink-3)',
        fontFamily: 'var(--ff-mono)',
      }}>
        v2.0.0-rc1 · scaffold + design tokens · components land in tasks 16-20
      </p>
      <div style={{
        marginTop: 'var(--s-5)',
        padding: 'var(--s-4)',
        background: 'var(--bg-elev)',
        boxShadow: 'var(--elev-1)',
        color: 'var(--ink-2)',
      }}>
        Token preview: warm cream <code style={{ fontFamily: 'var(--ff-mono)' }}>#FBFAF6</code> page,
        accent <span style={{ color: 'var(--acc)' }}>navy #2A4365</span>,
        deep ink <span style={{ color: 'var(--ink-1)', fontWeight: 600 }}>#15110A</span>.
      </div>
    </div>
  );
}
