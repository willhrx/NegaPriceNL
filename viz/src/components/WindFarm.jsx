import { useMemo } from 'react';
import { motion } from 'framer-motion';

/**
 * Get sky gradient based on hour of day
 */
function getSkyGradient(hour) {
  // Night (0-5, 21-23)
  if (hour < 6 || hour >= 21) {
    return ['#0a1628', '#0f1f3a'];
  }
  // Sunrise (6-8)
  if (hour < 9) {
    return ['#1e3a5f', '#f97316', '#fbbf24'];
  }
  // Day (9-17)
  if (hour < 18) {
    return ['#3b82f6', '#87CEEB'];
  }
  // Sunset (18-20)
  return ['#1e3a5f', '#f97316', '#dc2626'];
}

/**
 * Single wind turbine SVG with rotating blades
 */
function Turbine({ x, y, scale = 1, windSpeed }) {
  // Calculate rotation duration based on wind speed
  // Faster wind = faster rotation
  const rotationDuration = Math.max(0.5, 8 / (windSpeed + 1));

  const towerHeight = 80 * scale;
  const bladeLength = 40 * scale;
  const hubY = y - towerHeight;

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Tower */}
      <line
        x1="0"
        y1="0"
        x2="0"
        y2={-towerHeight}
        stroke="#475569"
        strokeWidth={4 * scale}
      />

      {/* Base */}
      <rect
        x={-15 * scale}
        y={-5}
        width={30 * scale}
        height={10 * scale}
        fill="#334155"
        rx={2}
      />

      {/* Hub */}
      <circle
        cx="0"
        cy={-towerHeight}
        r={6 * scale}
        fill="#64748b"
      />

      {/* Rotating blades */}
      <g
        style={{
          transformOrigin: `0px ${-towerHeight}px`,
          animation: `spin ${rotationDuration}s linear infinite`,
        }}
      >
        {[0, 120, 240].map((angle) => (
          <line
            key={angle}
            x1="0"
            y1={-towerHeight}
            x2={Math.sin((angle * Math.PI) / 180) * bladeLength}
            y2={-towerHeight - Math.cos((angle * Math.PI) / 180) * bladeLength}
            stroke="white"
            strokeWidth={3 * scale}
            strokeLinecap="round"
          />
        ))}
      </g>
    </g>
  );
}

/**
 * Cloud SVG
 */
function CloudShape({ x, y, scale = 1, opacity = 0.5 }) {
  return (
    <g transform={`translate(${x}, ${y}) scale(${scale})`} opacity={opacity}>
      <ellipse cx="25" cy="20" rx="25" ry="15" fill="white" />
      <ellipse cx="50" cy="25" rx="20" ry="12" fill="white" />
      <ellipse cx="40" cy="15" rx="15" ry="12" fill="white" />
    </g>
  );
}

export function WindFarm({ windSpeed, cloudCover, hour }) {
  const skyGradient = useMemo(() => getSkyGradient(hour), [hour]);
  const gradientId = `sky-gradient-${hour}`;

  // Cloud opacity based on cloud cover percentage
  const cloudOpacity = cloudCover / 100;

  // Turbine positions
  const turbines = [
    { x: 120, y: 280, scale: 0.8 },
    { x: 280, y: 260, scale: 1 },
    { x: 440, y: 270, scale: 0.9 },
    { x: 600, y: 265, scale: 0.95 },
    { x: 750, y: 275, scale: 0.85 },
  ];

  return (
    <div className="relative w-full h-full overflow-hidden rounded-xl">
      <svg
        viewBox="0 0 900 400"
        preserveAspectRatio="xMidYMid slice"
        className="w-full h-full"
      >
        {/* Sky gradient */}
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
            {skyGradient.map((color, i) => (
              <stop
                key={i}
                offset={`${(i / (skyGradient.length - 1)) * 100}%`}
                stopColor={color}
              />
            ))}
          </linearGradient>
        </defs>

        {/* Sky background */}
        <motion.rect
          x="0"
          y="0"
          width="900"
          height="400"
          fill={`url(#${gradientId})`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
        />

        {/* Clouds */}
        {cloudOpacity > 0.1 && (
          <>
            <CloudShape x={100} y={50} scale={1.2} opacity={cloudOpacity * 0.8} />
            <CloudShape x={350} y={30} scale={0.9} opacity={cloudOpacity * 0.6} />
            <CloudShape x={600} y={60} scale={1.1} opacity={cloudOpacity * 0.7} />
            <CloudShape x={750} y={40} scale={0.8} opacity={cloudOpacity * 0.5} />
          </>
        )}

        {/* Stars (only at night) */}
        {(hour < 6 || hour >= 21) && (
          <g opacity={0.6}>
            {[
              [120, 40], [250, 80], [400, 50], [550, 90], [700, 60],
              [80, 100], [320, 30], [480, 70], [620, 40], [800, 85]
            ].map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={1.5} fill="white" />
            ))}
          </g>
        )}

        {/* Ground / horizon */}
        <rect x="0" y="300" width="900" height="100" fill="#0f172a" />

        {/* Ground detail - grass/field line */}
        <path
          d="M0 300 Q225 290 450 300 Q675 310 900 300"
          fill="none"
          stroke="#1e293b"
          strokeWidth="2"
        />

        {/* Wind turbines */}
        {turbines.map((t, i) => (
          <Turbine
            key={i}
            x={t.x}
            y={t.y}
            scale={t.scale}
            windSpeed={windSpeed}
          />
        ))}

        {/* NordWind BV branding */}
        <text
          x="30"
          y="370"
          fill="#475569"
          fontSize="14"
          fontFamily="system-ui"
        >
          NordWind BV - 50 MW Wind Farm
        </text>
      </svg>
    </div>
  );
}
