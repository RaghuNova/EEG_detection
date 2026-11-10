import React from 'react';

export default function StatCard({ title, value, subtitle, icon: Icon, color = 'cyan', trend }) {
  const colorMap = {
    cyan: 'text-cyan-accent bg-cyan-accent/10 border-cyan-accent/30',
    blue: 'text-blue-accent bg-blue-accent/10 border-blue-accent/30',
    purple: 'text-purple-accent bg-purple-accent/10 border-purple-accent/30',
    emerald: 'text-emerald-accent bg-emerald-accent/10 border-emerald-accent/30',
    amber: 'text-amber-accent bg-amber-accent/10 border-amber-accent/30',
    rose: 'text-rose-accent bg-rose-accent/10 border-rose-accent/30',
  };

  return (
    <div className="p-4 rounded-xl bg-background-panel border border-border shadow-sm flex flex-col justify-between hover:border-text-muted/40 transition-all duration-200">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-text-muted">{title}</span>
        {Icon && (
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center border ${colorMap[color] || colorMap.cyan}`}>
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="mt-3">
        <div className="text-2xl font-bold tracking-tight text-text-primary font-mono">{value}</div>
        {subtitle && <p className="text-[11px] text-text-muted mt-0.5">{subtitle}</p>}
      </div>

      {trend && (
        <div className="mt-2 pt-2 border-t border-border/50 text-[10px] text-text-secondary">
          {trend}
        </div>
      )}
    </div>
  );
}
