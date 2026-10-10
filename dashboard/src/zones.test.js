import test from 'node:test';
import assert from 'node:assert/strict';
import { zoneActive } from './zones.js';

test('overnight schedules follow the UTC start weekday and exclude the end', () => {
  const zone = { armed: true, weekly: [{ weekdays: [4], start: '22:00:00', end: '06:00:00' }] };
  assert.equal(zoneActive(zone, new Date('2026-10-09T22:00:00Z')), true);
  assert.equal(zoneActive(zone, new Date('2026-10-10T04:00:00+05:30')), true);
  assert.equal(zoneActive(zone, new Date('2026-10-10T06:00:00Z')), false);
  assert.equal(zoneActive(zone, new Date('2026-10-10T22:00:00Z')), false);
});

test('manual disarm and absolute limits take precedence', () => {
  const zone = { armed: true, starts_at: '2026-10-10T08:00:00Z', ends_at: '2026-10-10T09:00:00Z' };
  assert.equal(zoneActive(zone, new Date('2026-10-10T08:30:00Z')), true);
  assert.equal(zoneActive(zone, new Date('2026-10-10T09:00:00Z')), false);
  assert.equal(zoneActive({ ...zone, armed: false }, new Date('2026-10-10T08:30:00Z')), false);
});
