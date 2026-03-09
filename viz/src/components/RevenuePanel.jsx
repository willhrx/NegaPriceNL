import { motion } from 'framer-motion';
import { Wind, Battery, Wallet, TrendingUp } from 'lucide-react';

/**
 * Format currency with Euro formatting
 */
function formatEuro(value) {
  return new Intl.NumberFormat('de-DE', {
    style: 'currency',
    currency: 'EUR',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

/**
 * Animated counter for revenue values
 */
function RevenueCounter({ value, label, icon: Icon, colorClass = 'text-white' }) {
  const isPositive = value >= 0;

  return (
    <div className="flex items-start gap-3">
      <div className={`p-2 rounded-lg bg-[#0a1628] ${colorClass}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <motion.div
          key={Math.floor(value)}
          className={`text-xl font-bold counter-value ${
            !isPositive ? 'text-red-400' : colorClass
          }`}
          initial={{ y: -5, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.1 }}
        >
          {formatEuro(value)}
        </motion.div>
        <div className="text-xs text-slate-500">{label}</div>
      </div>
    </div>
  );
}

export function RevenuePanel({ cumulativeRevenue, todayPnl }) {
  return (
    <div className="bg-[#162442] rounded-xl border border-[#1e3a5f] p-4">
      <div className="flex items-center gap-2 mb-4">
        <Wallet className="w-5 h-5 text-slate-400" />
        <span className="text-sm font-medium text-slate-400">Revenue</span>
      </div>

      {/* Cumulative totals */}
      <div className="space-y-4 mb-6">
        <RevenueCounter
          value={cumulativeRevenue.wind}
          label="Wind Revenue"
          icon={Wind}
          colorClass="text-blue-400"
        />
        <RevenueCounter
          value={cumulativeRevenue.bess}
          label="BESS P&L"
          icon={Battery}
          colorClass={cumulativeRevenue.bess >= 0 ? 'text-green-400' : 'text-red-400'}
        />

        {/* Total line */}
        <div className="pt-3 border-t border-[#1e3a5f]">
          <RevenueCounter
            value={cumulativeRevenue.total}
            label="Total Portfolio"
            icon={TrendingUp}
            colorClass="text-emerald-400"
          />
        </div>
      </div>

      {/* Today's P&L */}
      <div className="pt-3 border-t border-[#1e3a5f]">
        <div className="text-xs text-slate-500 mb-2">Today's P&L</div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-slate-400">Wind: </span>
            <span className="text-blue-400 counter-value">
              {formatEuro(todayPnl.wind)}
            </span>
          </div>
          <div>
            <span className="text-slate-400">BESS: </span>
            <span className={`counter-value ${todayPnl.bess >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {formatEuro(todayPnl.bess)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
