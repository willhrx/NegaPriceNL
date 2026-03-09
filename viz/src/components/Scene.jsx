import { WindFarm } from './WindFarm';
import { BatteryGauge } from './BatteryGauge';
import { PriceTicker } from './PriceTicker';
import { RevenuePanel } from './RevenuePanel';
import { WeatherOverlay } from './WeatherOverlay';
import { DayChart } from './DayChart';
import { TimelineControls } from './TimelineControls';
import { Wind } from 'lucide-react';

export function Scene({
  simulation,
}) {
  const {
    currentDay,
    currentMtu,
    dayIndex,
    mtuIndex,
    totalDays,
    isPlaying,
    speed,
    cumulativeRevenue,
    todayPnl,
    currentTimestamp,
    togglePlay,
    prevDay,
    nextDay,
    jumpToDay,
    jumpToMtu,
    setPlaybackSpeed,
  } = simulation;

  // Loading state
  if (!currentDay || !currentMtu) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0a1628]">
        <div className="text-center">
          <div className="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
          <div className="text-lg text-slate-400">Loading simulation data...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-[#0a1628]">
      {/* Header */}
      <header className="bg-[#0f1f3a] border-b border-[#1e3a5f] px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Wind className="w-8 h-8 text-blue-400" />
            <div>
              <h1 className="text-xl font-bold text-white">NegaPriceNL</h1>
              <p className="text-xs text-slate-500">BESS Arbitrage Simulation</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-lg font-semibold text-white">{currentTimestamp}</div>
            <div className="text-xs text-slate-500">Day {dayIndex + 1} of {totalDays}</div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 p-4 overflow-hidden">
        <div className="grid grid-cols-12 gap-4 h-full">
          {/* Left side - Wind Farm Scene */}
          <div className="col-span-7 flex flex-col gap-4">
            {/* Wind farm visualization */}
            <div className="flex-1 min-h-0">
              <WindFarm
                windSpeed={currentMtu.wind_speed_ms}
                cloudCover={currentMtu.cloud_cover_pct}
                hour={currentMtu.hour}
              />
            </div>

            {/* Day chart */}
            <div className="h-[200px]">
              <DayChart
                mtus={currentDay.mtus}
                currentMtuIndex={mtuIndex}
              />
            </div>
          </div>

          {/* Right side - Stats panels */}
          <div className="col-span-5 flex flex-col gap-4">
            {/* Price ticker */}
            <PriceTicker
              price={currentMtu.price_eur_mwh}
              timestamp={currentTimestamp}
            />

            {/* Battery gauge */}
            <BatteryGauge
              socMwh={currentMtu.soc_mwh}
              chargeMw={currentMtu.charge_mw}
              dischargeMw={currentMtu.discharge_mw}
            />

            {/* Weather overlay */}
            <WeatherOverlay
              windSpeed={currentMtu.wind_speed_ms}
              temperature={currentMtu.temperature_c}
              cloudCover={currentMtu.cloud_cover_pct}
              windGeneration={currentMtu.wind_generation_mw}
            />

            {/* Revenue panel */}
            <div className="flex-1 min-h-0">
              <RevenuePanel
                cumulativeRevenue={cumulativeRevenue}
                todayPnl={todayPnl}
              />
            </div>
          </div>
        </div>
      </main>

      {/* Timeline controls */}
      <TimelineControls
        isPlaying={isPlaying}
        speed={speed}
        dayIndex={dayIndex}
        mtuIndex={mtuIndex}
        totalDays={totalDays}
        currentTimestamp={currentTimestamp}
        onTogglePlay={togglePlay}
        onPrevDay={prevDay}
        onNextDay={nextDay}
        onJumpToDay={jumpToDay}
        onJumpToMtu={jumpToMtu}
        onSetSpeed={setPlaybackSpeed}
      />
    </div>
  );
}
