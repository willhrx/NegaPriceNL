import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight } from 'lucide-react';

const SPEED_OPTIONS = [1, 4, 10, 50];

export function TimelineControls({
  isPlaying,
  speed,
  dayIndex,
  mtuIndex,
  totalDays,
  currentTimestamp,
  onTogglePlay,
  onPrevDay,
  onNextDay,
  onJumpToDay,
  onJumpToMtu,
  onSetSpeed,
}) {
  return (
    <div className="bg-[#0f1f3a] border-t border-[#1e3a5f] px-6 py-4">
      <div className="flex items-center justify-between gap-6">
        {/* Playback controls */}
        <div className="flex items-center gap-2">
          {/* Previous day */}
          <button
            onClick={onPrevDay}
            disabled={dayIndex === 0}
            className="p-2 rounded-lg bg-[#162442] hover:bg-[#1e3a5f] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            title="Previous day"
          >
            <SkipBack className="w-5 h-5" />
          </button>

          {/* Play/Pause */}
          <button
            onClick={onTogglePlay}
            className="p-3 rounded-full bg-blue-600 hover:bg-blue-500 transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <Pause className="w-6 h-6" />
            ) : (
              <Play className="w-6 h-6 ml-0.5" />
            )}
          </button>

          {/* Next day */}
          <button
            onClick={onNextDay}
            disabled={dayIndex >= totalDays - 1}
            className="p-2 rounded-lg bg-[#162442] hover:bg-[#1e3a5f] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            title="Next day"
          >
            <SkipForward className="w-5 h-5" />
          </button>
        </div>

        {/* Speed selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-400">Speed:</span>
          <div className="flex bg-[#162442] rounded-lg p-1">
            {SPEED_OPTIONS.map((s) => (
              <button
                key={s}
                onClick={() => onSetSpeed(s)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  speed === s
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>

        {/* Day scrubber */}
        <div className="flex-1 flex items-center gap-4">
          <span className="text-sm text-slate-400 whitespace-nowrap">
            Day {dayIndex + 1}/{totalDays}
          </span>
          <input
            type="range"
            min="0"
            max={totalDays - 1}
            value={dayIndex}
            onChange={(e) => onJumpToDay(parseInt(e.target.value))}
            className="flex-1 h-2 bg-[#162442] rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        {/* MTU scrubber */}
        <div className="flex items-center gap-4">
          <span className="text-sm text-slate-400 whitespace-nowrap">
            MTU {mtuIndex + 1}/96
          </span>
          <input
            type="range"
            min="0"
            max="95"
            value={mtuIndex}
            onChange={(e) => onJumpToMtu(parseInt(e.target.value))}
            className="w-32 h-2 bg-[#162442] rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        {/* Current timestamp */}
        <div className="text-right">
          <div className="text-lg font-semibold text-white counter-value">
            {currentTimestamp}
          </div>
          <div className="text-xs text-slate-500">CET</div>
        </div>
      </div>
    </div>
  );
}
