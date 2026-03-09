import { useSimulation } from './hooks/useSimulation';
import { Scene } from './components/Scene';

function App() {
  const simulation = useSimulation(true); // Use sample data

  if (simulation.loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0a1628]">
        <div className="text-center">
          <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <div className="text-lg text-slate-400">Loading simulation data...</div>
        </div>
      </div>
    );
  }

  if (simulation.error) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0a1628]">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">Error loading data</div>
          <div className="text-slate-400">{simulation.error}</div>
        </div>
      </div>
    );
  }

  return <Scene simulation={simulation} />;
}

export default App;
