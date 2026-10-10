import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';
import { zoneActive } from './zones.js';

const api = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const formatTime = value => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const label = value => value.replaceAll('_', ' ');

function App() {
  const [key, setKey] = useState('');
  const [draft, setDraft] = useState('');
  const [snapshot, setSnapshot] = useState({ rooms: [], events: [], devices: [], zones: [] });
  const [error, setError] = useState('');
  const [updated, setUpdated] = useState(null);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    if (!key) return;
    let active = true;
    let timer;
    const controller = new AbortController();
    async function refresh() {
      try {
        const paths = ['rooms/status', 'events', 'devices', 'zones'];
        const values = await Promise.all(paths.map(async path => {
          const res = await fetch(`${api}/api/${path}`, { headers: { 'X-API-Key': key }, signal: controller.signal });
          if (!res.ok) throw new Error(res.status === 401 ? 'Read key rejected. Reconnect with READ_API_KEY.' : `API returned ${res.status}. Check the backend and database.`);
          return res.json();
        }));
        if (active) {
          setSnapshot({ rooms: values[0], events: values[1], devices: values[2], zones: values[3] });
          setUpdated(new Date()); setError('');
        }
      } catch (err) {
        if (active && err.name !== 'AbortError') setError(err.message || 'API unavailable');
      } finally {
        if (active) timer = setTimeout(refresh, 3000);
      }
    }
    refresh();
    return () => { active = false; controller.abort(); clearTimeout(timer); };
  }, [key]);

  const events = snapshot.events.filter(e => filter === 'all' || e.type === filter);
  const disconnect = () => { setKey(''); setDraft(''); setUpdated(null); setError(''); setSnapshot({ rooms: [], events: [], devices: [], zones: [] }); };

  return <div className="app">
    <aside>
      <a className="brand" href="#overview"><span className="mark">◉</span> sentinel<span className="brand-dot">.</span></a>
      <div className="workspace">PERSONAL WORKSPACE <span>01</span></div>
      <nav aria-label="Workspace"><a className="active" href="#overview">▦ <span>Overview</span></a><a href="#timeline">≋ <span>Event timeline</span></a><a href="#devices">⌁ <span>Devices</span></a></nav>
      <div className="privacy"><span>◎</span><strong>Presence without pictures.</strong><p>Local signal processing.<br/>Identity disabled.<br/>Only events leave the gateway.</p></div>
      <div className="version">M2 / RESEARCH PREVIEW</div>
    </aside>
    <main id="overview">
      <header><div className="breadcrumb">Workspace <span>/</span> Overview</div><span className={'status ' + (key && !error && updated ? 'live' : '')}><i/>{key && !error && updated ? 'Polling connected' : 'Not connected'}</span></header>
      <section className="intro"><div><div className="eyebrow">WI-FI CSI SENSING PLATFORM</div><h1>A quieter way to sense.</h1><p>Your space, understood through signals.</p></div><span className="mode">RESEARCH PREVIEW</span></section>
      <div className="notice"><span>i</span><p>Calibrated heuristic detection. Occupancy is an estimate; fall alerts and room-zone transitions still need real-world validation.</p></div>
      {!key ? <form className="connect" onSubmit={e => { e.preventDefault(); setKey(draft.trim()); setDraft(''); }}>
        <div><h2>Connect your workspace</h2><p>Start the API and gateway, then enter your read-only dashboard key. It stays in this tab’s memory.</p></div>
        <label>Dashboard read key<input required type="password" value={draft} onChange={e => setDraft(e.target.value)} placeholder="READ_API_KEY" autoComplete="off"/></label><button type="submit">Connect →</button>
      </form> : <div className="connection"><span>{updated ? `Last refreshed ${formatTime(updated)}` : 'Connecting…'} · refreshes every 3 seconds</span><button className="text-button" onClick={disconnect}>Disconnect</button></div>}
      {error && <div role="alert" className="error">{error} Previously loaded values may be outdated.</div>}
      <section className="metrics" aria-label="Workspace metrics">
        <div><span>ROOMS REPORTING</span><strong>{snapshot.rooms.length.toString().padStart(2, '0')}</strong><small>Latest presence / occupancy events</small></div>
        <div><span>ACTIVE SIGNALS</span><strong>{snapshot.devices.filter(d => d.health === 'online').length.toString().padStart(2, '0')}</strong><small>Source event within 30 seconds</small></div>
        <div><span>ARMED ZONES</span><strong>{snapshot.zones.filter(z => zoneActive(z)).length.toString().padStart(2, '0')}</strong><small>Active UTC intervals and weekly schedules</small></div>
      </section>
      <section className="section"><div className="section-heading"><h2>Room overview</h2><span>01 / LIVE STATE</span></div>
        <div className="rooms">{snapshot.rooms.length ? snapshot.rooms.map(room => <article className="room" key={room.room_id}>
          <div className="room-top"><span className="room-icon">▧</span><span className={'pill ' + (room.stale || error ? '' : 'green')}>{room.stale || error ? 'Stale / unknown' : room.payload.occupied === true ? 'Occupied' : room.payload.occupied === false ? 'Empty' : 'Unknown'}</span></div>
          <h3>{label(room.room_id)}</h3><p>Camera-free sensing · {room.payload.source || 'event source'}</p><div className="signal-art" aria-hidden="true">{Array.from({ length: 30 }, (_, i) => <i key={i} style={{ height: `${10 + Math.sin(i * 1.7) ** 2 * (room.payload.occupied ? 38 : 9)}px` }}/>)}</div>
          <footer><span>Source time {formatTime(room.ts)}</span><span>{room.payload.confidence_kind ? 'Heuristic · not a probability' : `Confidence ${Math.round(room.confidence * 100)}%`}</span></footer>
        </article>) : <div className="empty"><span>⌁</span><h3>Waiting for your first signal</h3><p>Calibrate the room, then run a replay to populate room state and events.</p></div>}</div>
      </section>
      <div className="lower"><section className="section timeline" id="timeline"><div className="section-heading"><h2>Event timeline</h2><select aria-label="Filter event type" value={filter} onChange={e => setFilter(e.target.value)}><option value="all">All events</option>{[...new Set(snapshot.events.map(e => e.type))].sort().map(type => <option key={type} value={type}>{label(type)}</option>)}</select></div>
        {events.length ? <div className="event-list">{events.map(event => <article className="event" key={event.event_id}><span className={'event-dot ' + (event.type.includes('fall') ? 'amber' : '')}/><div><strong>{label(event.type)}{event.payload.status ? ` · ${label(event.payload.status)}` : ''}</strong><p>{event.room_id} · {event.device_id} · seq {event.seq}</p></div><time dateTime={event.ts}>{formatTime(event.ts)}</time></article>)}</div> : <p className="empty-small">No events to display yet.</p>}
      </section><section className="section" id="devices"><div className="section-heading"><h2>Device health</h2><span>02 / GATEWAY</span></div>{snapshot.devices.length ? snapshot.devices.map(device => <article className="device" key={device.device_id}><div><span className="device-icon">⌁</span><strong>{device.device_id}</strong></div><span className={'pill ' + (device.health === 'online' && !error ? 'green' : '')}>{error ? 'unknown' : device.health}</span><p>Last source event {formatTime(device.last_event_at)}<br/>Latest sequence {device.last_seq}</p></article>) : <p className="empty-small">Devices appear after their first accepted event.</p>}<div className="pipeline">ESP32 <span>→</span> Local gateway <span>→</span> API<p>Serial adapter pending real hardware verification.</p></div></section></div>
      <footer className="page-footer"><span>SENTINEL / Built for curiosity. Designed for privacy.</span><span>Times shown in your browser’s timezone · API uses UTC</span></footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
