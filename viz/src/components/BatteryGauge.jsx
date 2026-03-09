import { motion } from 'framer-motion';
import { Zap, ArrowDown, ArrowUp } from 'lucide-react';

/**
 * Get fill color based on SoC percentage
 */
function getFillColor(pct) {
  if (pct < 20) return '#ef4444'; // red
  if (pct < 60) return '#eab308'; // yellow
  return '#22c55e'; // green
}

export function BatteryGauge({
  socMwh,
  chargeMw,
  dischargeMw,
  maxEnergy = 50,
  minSoc = 5,
  maxSoc = 45,
}) {
  // Calculate fill percentage (usable range: 5-45 MWh)
  const usableRange = maxSoc - minSoc;
  const fillPct = Math.max(0, Math.min(100, ((socMwh - minSoc) / usableRange) * 100));

  const fillColor = getFillColor(fillPct);
  const isCharging = chargeMw > 0;
  const isDischarging = dischargeMw > 0;

  return (
    <div className="bg-[#162442] rounded-xl border border-[#1e3a5f] p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-slate-400">Battery Storage</span>
        <span className="text-xs text-slate-500">25 MW / 50 MWh</span>
      </div>

      {/* Battery visual */}
      <div className="relative">
        {/* Battery outline */}
        <div className="relative w-full h-24 bg-[#0a1628] rounded-lg border-2 border-[#1e3a5f] overflow-hidden">
          {/* Battery cap */}
          <div className="absolute -right-1 top-1/2 -translate-y-1/2 w-3 h-8 bg-[#1e3a5f] rounded-r-md" />

          {/* Fill level */}
          <motion.div
            className="absolute bottom-0 left-0 right-0 rounded-b-md"
            style={{ backgroundColor: fillColor }}
            initial={{ height: 0 }}
            animate={{ height: `${fillPct}%` }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
          />

          {/* Charging indicator overlay */}
          {isCharging && (
            <motion.div
              className="absolute inset-0 bg-blue-500/20 flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0.2, 0.4, 0.2] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              <ArrowDown className="w-8 h-8 text-blue-400" />
            </motion.div>
          )}

          {/* Discharging indicator overlay */}
          {isDischarging && (
            <motion.div
              className="absolute inset-0 bg-orange-500/20 flex items-center justify-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0.2, 0.4, 0.2] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              <ArrowUp className="w-8 h-8 text-orange-400" />
            </motion.div>
          )}

          {/* SoC percentage text */}
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-2xl font-bold text-white drop-shadow-lg">
              {fillPct.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="mt-3 grid grid-cols-2 gap-2">
        {/* SoC in MWh */}
        <div className="text-center">
          <div className="text-lg font-semibold text-white counter-value">
            {socMwh.toFixed(1)} MWh
          </div>
          <div className="text-xs text-slate-500">State of Charge</div>
        </div>

        {/* Power flow */}
        <div className="text-center">
          {isCharging && (
            <div className="flex items-center justify-center gap-1">
              <Zap className="w-4 h-4 text-blue-400" />
              <span className="text-lg font-semibold text-blue-400">
                +{chargeMw.toFixed(1)} MW
              </span>
            </div>
          )}
          {isDischarging && (
            <div className="flex items-center justify-center gap-1">
              <Zap className="w-4 h-4 text-orange-400" />
              <span className="text-lg font-semibold text-orange-400">
                -{dischargeMw.toFixed(1)} MW
              </span>
            </div>
          )}
          {!isCharging && !isDischarging && (
            <div className="text-lg font-semibold text-slate-500">Idle</div>
          )}
          <div className="text-xs text-slate-500">Power Flow</div>
        </div>
      </div>
    </div>
  );
}
