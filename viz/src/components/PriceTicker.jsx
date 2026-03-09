import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';

/**
 * Get price color class based on price value
 */
function getPriceColor(price) {
  if (price < 0) return 'text-red-500';
  if (price < 20) return 'text-orange-400';
  if (price > 100) return 'text-green-400';
  return 'text-white';
}

/**
 * Get background color class based on price value
 */
function getPriceBackground(price) {
  if (price < 0) return 'bg-red-900/30 border-red-500/50';
  if (price < 20) return 'bg-orange-900/20 border-orange-500/30';
  if (price > 100) return 'bg-green-900/20 border-green-500/30';
  return 'bg-[#162442] border-[#1e3a5f]';
}

export function PriceTicker({ price, timestamp }) {
  const isNegative = price < 0;
  const priceColor = getPriceColor(price);
  const bgClass = getPriceBackground(price);

  return (
    <motion.div
      className={`rounded-xl border-2 p-4 ${bgClass} ${isNegative ? 'negative-price-flash' : ''}`}
      initial={{ scale: 1 }}
      animate={{ scale: isNegative ? [1, 1.02, 1] : 1 }}
      transition={{ duration: 0.5, repeat: isNegative ? Infinity : 0 }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-slate-400">Day-Ahead Price</span>
        {isNegative && (
          <span className="flex items-center gap-1 text-xs text-red-400 bg-red-900/50 px-2 py-0.5 rounded-full">
            <AlertTriangle className="w-3 h-3" />
            NEGATIVE
          </span>
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <motion.span
          key={price}
          className={`text-4xl font-bold counter-value ${priceColor}`}
          initial={{ y: -10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.15 }}
        >
          {price >= 0 ? '' : ''}
          {price.toFixed(2)}
        </motion.span>
        <span className="text-lg text-slate-400">/MWh</span>
      </div>

      <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
        {price > 50 ? (
          <TrendingUp className="w-3 h-3 text-green-400" />
        ) : (
          <TrendingDown className="w-3 h-3 text-red-400" />
        )}
        <span>Quarter-hourly clearing price</span>
      </div>
    </motion.div>
  );
}
