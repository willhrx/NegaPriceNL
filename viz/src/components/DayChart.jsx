import { useMemo } from 'react';
import {
  ComposedChart,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';

/**
 * Custom tooltip for the chart
 */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;

  const data = payload[0].payload;
  const hour = Math.floor(label / 4);
  const minute = (label % 4) * 15;
  const timeStr = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

  return (
    <div className="bg-[#0f1f3a] border border-[#1e3a5f] rounded-lg p-3 shadow-xl">
      <div className="text-sm text-slate-400 mb-1">{timeStr}</div>
      <div className="text-lg font-bold text-white">
        {data.price.toFixed(2)} /MWh
      </div>
      {data.charge > 0 && (
        <div className="text-sm text-blue-400">Charge: {data.charge.toFixed(1)} MW</div>
      )}
      {data.discharge > 0 && (
        <div className="text-sm text-orange-400">Discharge: {data.discharge.toFixed(1)} MW</div>
      )}
    </div>
  );
}

export function DayChart({ mtus, currentMtuIndex }) {
  // Transform MTU data for the chart
  const chartData = useMemo(() => {
    if (!mtus) return [];

    return mtus.map((mtu, i) => ({
      t: i,
      price: mtu.price_eur_mwh,
      charge: mtu.charge_mw,
      discharge: mtu.discharge_mw,
      // For coloring the area
      pricePositive: mtu.price_eur_mwh >= 0 ? mtu.price_eur_mwh : 0,
      priceNegative: mtu.price_eur_mwh < 0 ? mtu.price_eur_mwh : 0,
    }));
  }, [mtus]);

  // Calculate Y-axis domain
  const [minPrice, maxPrice] = useMemo(() => {
    if (!mtus || mtus.length === 0) return [-50, 150];
    const prices = mtus.map(m => m.price_eur_mwh);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    return [Math.floor(min / 20) * 20 - 20, Math.ceil(max / 20) * 20 + 20];
  }, [mtus]);

  // X-axis tick formatter (show hours)
  const xAxisFormatter = (value) => {
    if (value % 4 === 0) {
      return Math.floor(value / 4).toString();
    }
    return '';
  };

  return (
    <div className="bg-[#162442] rounded-xl border border-[#1e3a5f] p-4 h-full">
      <div className="text-sm font-medium text-slate-400 mb-2">Today's Price Profile</div>

      <ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="negativeGradient" x1="0" y1="1" x2="0" y2="0">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
          </defs>

          <XAxis
            dataKey="t"
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickFormatter={xAxisFormatter}
            interval={3}
          />
          <YAxis
            stroke="#475569"
            tick={{ fill: '#64748b', fontSize: 10 }}
            domain={[minPrice, maxPrice]}
            width={40}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Zero line */}
          <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />

          {/* Current MTU marker */}
          <ReferenceLine
            x={currentMtuIndex}
            stroke="#fbbf24"
            strokeWidth={2}
          />

          {/* Price area - positive */}
          <Area
            type="monotone"
            dataKey="pricePositive"
            stroke="#3b82f6"
            fill="url(#priceGradient)"
            strokeWidth={1.5}
          />

          {/* Price area - negative */}
          <Area
            type="monotone"
            dataKey="priceNegative"
            stroke="#ef4444"
            fill="url(#negativeGradient)"
            strokeWidth={1.5}
          />

          {/* Charge bars (downward) */}
          <Bar
            dataKey="charge"
            fill="#3b82f6"
            opacity={0.6}
            maxBarSize={8}
          />

          {/* Discharge bars (on secondary axis concept - just overlay) */}
          <Bar
            dataKey="discharge"
            fill="#f97316"
            opacity={0.6}
            maxBarSize={8}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="flex justify-between text-xs text-slate-500 mt-1">
        <span>0h</span>
        <span>6h</span>
        <span>12h</span>
        <span>18h</span>
        <span>24h</span>
      </div>
    </div>
  );
}
