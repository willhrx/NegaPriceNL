import { Wind, Thermometer, Cloud, Zap } from 'lucide-react';

export function WeatherOverlay({
  windSpeed,
  temperature,
  cloudCover,
  windGeneration,
  windCapacity = 50,
}) {
  const capacityFactor = ((windGeneration / windCapacity) * 100).toFixed(0);

  return (
    <div className="bg-[#162442]/90 backdrop-blur-sm rounded-xl border border-[#1e3a5f] p-4">
      <div className="text-sm font-medium text-slate-400 mb-3">Weather & Generation</div>

      <div className="grid grid-cols-2 gap-4">
        {/* Wind speed */}
        <div className="flex items-center gap-2">
          <Wind className="w-5 h-5 text-blue-400" />
          <div>
            <div className="text-lg font-semibold text-white">
              {windSpeed.toFixed(1)} m/s
            </div>
            <div className="text-xs text-slate-500">Wind Speed</div>
          </div>
        </div>

        {/* Temperature */}
        <div className="flex items-center gap-2">
          <Thermometer className="w-5 h-5 text-orange-400" />
          <div>
            <div className="text-lg font-semibold text-white">
              {temperature.toFixed(1)}°C
            </div>
            <div className="text-xs text-slate-500">Temperature</div>
          </div>
        </div>

        {/* Cloud cover */}
        <div className="flex items-center gap-2">
          <Cloud className="w-5 h-5 text-slate-400" />
          <div>
            <div className="text-lg font-semibold text-white">
              {cloudCover.toFixed(0)}%
            </div>
            <div className="text-xs text-slate-500">Cloud Cover</div>
          </div>
        </div>

        {/* Wind generation */}
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-green-400" />
          <div>
            <div className="text-lg font-semibold text-white">
              {windGeneration.toFixed(1)} MW
            </div>
            <div className="text-xs text-slate-500">
              Wind Gen ({capacityFactor}% CF)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
