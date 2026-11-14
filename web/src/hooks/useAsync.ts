import { useCallback, useEffect, useRef, useState } from "react";

export interface Async<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Load something when the component appears (and again every `every` ms, if given). */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[], every?: number): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const latest = useRef(0);

  const reload = useCallback(() => {
    const ticket = ++latest.current;
    load().then(
      (value) => {
        if (ticket !== latest.current) return; // a newer request has been made since
        setData(value);
        setError(null);
        setLoading(false);
      },
      (reason: unknown) => {
        if (ticket !== latest.current) return;
        setError(reason instanceof Error ? reason.message : String(reason));
        setLoading(false);
      },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    setLoading(true);
    reload();
    if (!every) return;
    const timer = window.setInterval(reload, every);
    return () => window.clearInterval(timer);
  }, [reload, every]);

  return { data, error, loading, reload };
}
