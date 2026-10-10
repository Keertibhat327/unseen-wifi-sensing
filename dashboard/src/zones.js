// The API contract uses UTC start weekdays and half-open intervals [start, end).
export function zoneActive(zone, now = new Date()) {
  if (!zone.armed || (zone.starts_at && now < new Date(zone.starts_at)) ||
      (zone.ends_at && now >= new Date(zone.ends_at))) return false;
  if (!zone.weekly?.length) return true;
  const weekday = (now.getUTCDay() + 6) % 7;
  const current = now.getUTCHours() * 3600 + now.getUTCMinutes() * 60 + now.getUTCSeconds();
  const seconds = text => {
    const [h, m, s = 0] = text.split(':').map(Number);
    return h * 3600 + m * 60 + s;
  };
  return zone.weekly.some(rule => {
    const start = seconds(rule.start), end = seconds(rule.end);
    return start < end
      ? rule.weekdays.includes(weekday) && start <= current && current < end
      : (rule.weekdays.includes(weekday) && current >= start) ||
        (rule.weekdays.includes((weekday + 6) % 7) && current < end);
  });
}
