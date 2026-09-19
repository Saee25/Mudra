export default function App() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-6">
      <div className="bg-cream-50 border border-cream-300 shadow-sm rounded-2xl p-12 max-w-lg w-full text-center space-y-6 flex flex-col items-center">
        <div className="bg-blush-100 text-blush-800 px-3 py-1 rounded-full text-sm font-medium tracking-wide uppercase">
          Coming Soon
        </div>
        
        <h1 className="text-6xl md:text-7xl font-display font-medium text-ink">
          Mudra
        </h1>
        
        <p className="text-ink-muted text-lg">
          Real-time Indian Sign Language recognition, right in your browser
        </p>
      </div>
    </main>
  );
}
