import { useState, useEffect, useCallback, useMemo } from 'react';

// Import sample data (full data can be loaded dynamically)
import sampleData from '../data/simulation_output_sample.json';

/**
 * Custom hook for BESS simulation playback
 *
 * Manages:
 * - Loading simulation data
 * - Playback state (day index, MTU index, playing, speed)
 * - Derived values (cumulative revenue, current weather, etc.)
 */
export function useSimulation(useSample = true) {
  // Data
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Playback state
  const [dayIndex, setDayIndex] = useState(0);
  const [mtuIndex, setMtuIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1); // 1 = 1 MTU per 100ms

  // Load data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        if (useSample) {
          setData(sampleData);
        } else {
          // Try to load full data, fall back to sample if not available
          try {
            const response = await fetch('/data/simulation_output.json');
            if (response.ok) {
              const fullData = await response.json();
              setData(fullData);
            } else {
              // Fall back to sample data
              console.warn('Full data not available, using sample data');
              setData(sampleData);
            }
          } catch {
            // Fall back to sample data
            console.warn('Could not load full data, using sample data');
            setData(sampleData);
          }
        }
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };
    loadData();
  }, [useSample]);

  // Current day and MTU
  const currentDay = data?.days?.[dayIndex] || null;
  const currentMtu = currentDay?.mtus?.[mtuIndex] || null;

  // Playback loop
  useEffect(() => {
    if (!isPlaying || !data) return;

    const interval = setInterval(() => {
      setMtuIndex(prev => {
        if (prev >= 95) {
          setDayIndex(d => {
            if (d >= data.days.length - 1) {
              setIsPlaying(false);
              return d;
            }
            return d + 1;
          });
          return 0;
        }
        return prev + 1;
      });
    }, 100 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, speed, data]);

  // Cumulative revenue calculation
  const cumulativeRevenue = useMemo(() => {
    if (!data || !currentDay) {
      return { wind: 0, bess: 0, total: 0 };
    }

    // Sum all previous days
    let windTotal = 0;
    let bessTotal = 0;

    for (let d = 0; d < dayIndex; d++) {
      const day = data.days[d];
      windTotal += day.daily_summary.wind_revenue_eur;
      bessTotal += day.daily_summary.bess_net_pnl_eur;
    }

    // Add current day up to current MTU (approximation)
    const mtusToSum = currentDay.mtus.slice(0, mtuIndex + 1);
    for (const mtu of mtusToSum) {
      // Wind revenue: generation * price * 0.25h
      windTotal += mtu.wind_generation_mw * mtu.price_eur_mwh * 0.25;
      // BESS revenue: discharge - charge
      bessTotal += (mtu.discharge_mw - mtu.charge_mw) * mtu.price_eur_mwh * 0.25;
    }

    return {
      wind: windTotal,
      bess: bessTotal,
      total: windTotal + bessTotal,
    };
  }, [data, dayIndex, mtuIndex, currentDay]);

  // Today's P&L (up to current MTU)
  const todayPnl = useMemo(() => {
    if (!currentDay) return { wind: 0, bess: 0, total: 0 };

    let windTotal = 0;
    let bessTotal = 0;

    const mtusToSum = currentDay.mtus.slice(0, mtuIndex + 1);
    for (const mtu of mtusToSum) {
      windTotal += mtu.wind_generation_mw * mtu.price_eur_mwh * 0.25;
      bessTotal += (mtu.discharge_mw - mtu.charge_mw) * mtu.price_eur_mwh * 0.25;
    }

    return {
      wind: windTotal,
      bess: bessTotal,
      total: windTotal + bessTotal,
    };
  }, [currentDay, mtuIndex]);

  // Playback controls
  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);
  const togglePlay = useCallback(() => setIsPlaying(prev => !prev), []);

  const nextDay = useCallback(() => {
    if (!data) return;
    setMtuIndex(0);
    setDayIndex(prev => Math.min(prev + 1, data.days.length - 1));
  }, [data]);

  const prevDay = useCallback(() => {
    setMtuIndex(0);
    setDayIndex(prev => Math.max(prev - 1, 0));
  }, []);

  const jumpToDay = useCallback((index) => {
    if (!data) return;
    setMtuIndex(0);
    setDayIndex(Math.max(0, Math.min(index, data.days.length - 1)));
  }, [data]);

  const jumpToMtu = useCallback((index) => {
    setMtuIndex(Math.max(0, Math.min(index, 95)));
  }, []);

  const setPlaybackSpeed = useCallback((newSpeed) => {
    setSpeed(newSpeed);
  }, []);

  const reset = useCallback(() => {
    setDayIndex(0);
    setMtuIndex(0);
    setIsPlaying(false);
  }, []);

  // Format current timestamp
  const currentTimestamp = useMemo(() => {
    if (!currentDay || !currentMtu) return '';
    const date = new Date(currentDay.date);
    const hour = currentMtu.hour.toString().padStart(2, '0');
    const minute = (currentMtu.quarter * 15).toString().padStart(2, '0');
    return `${date.toLocaleDateString('en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    })} ${hour}:${minute}`;
  }, [currentDay, currentMtu]);

  return {
    // Data
    data,
    loading,
    error,
    metadata: data?.metadata || null,

    // Current state
    dayIndex,
    mtuIndex,
    currentDay,
    currentMtu,
    totalDays: data?.days?.length || 0,

    // Playback state
    isPlaying,
    speed,

    // Derived values
    cumulativeRevenue,
    todayPnl,
    currentTimestamp,

    // Controls
    play,
    pause,
    togglePlay,
    nextDay,
    prevDay,
    jumpToDay,
    jumpToMtu,
    setPlaybackSpeed,
    reset,
  };
}
