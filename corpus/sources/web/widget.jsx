import React, { useCallback, useEffect, useMemo, useState } from 'react';

export function useDebounced(value, delay = 250) {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setSettled(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return settled;
}

export default function ServiceTable({ items, onSelect }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState({ key: 'name', dir: 'asc' });
  const debounced = useDebounced(query);

  const rows = useMemo(() => {
    const needle = debounced.trim().toLowerCase();
    const filtered = needle
      ? items.filter((it) => it.name.toLowerCase().includes(needle))
      : items.slice();
    const sign = sort.dir === 'asc' ? 1 : -1;
    return filtered.sort((a, b) => sign * String(a[sort.key]).localeCompare(String(b[sort.key])));
  }, [items, debounced, sort]);

  const toggleSort = useCallback((key) => {
    setSort((prev) => ({ key, dir: prev.key === key && prev.dir === 'asc' ? 'desc' : 'asc' }));
  }, []);

  if (!items.length) return <p className="empty">No services yet.</p>;

  return (
    <table className="grid">
      <thead>
        <tr>
          {['name', 'owner', 'state'].map((key) => (
            <th key={key} onClick={() => toggleSort(key)} aria-sort={sort.key === key ? sort.dir : 'none'}>
              {key}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((item) => (
          <tr key={item.name} onClick={() => onSelect?.(item)}>
            <td>{item.name}</td>
            <td>{item.owner}</td>
            <td><span className={`pill ${item.state}`}>{item.state}</span></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
