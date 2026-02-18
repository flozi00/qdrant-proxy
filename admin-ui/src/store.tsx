/* ============================================================================
 * Global application state via React Context
 *
 * Stores authentication, selected collection, and collection list.
 * ============================================================================ */

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { apiFetch, setAdminKey, validateKey } from './api/client';
import type { AdminStats, CollectionInfo } from './types';

interface AppState {
  adminKey: string;
  isLoggedIn: boolean;
  currentCollection: string;
  collections: CollectionInfo[];
  totalDocuments: number;
  totalFaqs: number;
}

interface AppActions {
  login: (key: string) => Promise<void>;
  logout: () => void;
  setCollection: (name: string) => void;
  refreshStats: () => Promise<void>;
}

const AppContext = createContext<(AppState & AppActions) | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [adminKey, setKey] = useState('');
  const [isLoggedIn, setLoggedIn] = useState(false);
  const [currentCollection, setCurrentCollection] = useState('');
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [totalFaqs, setTotalFaqs] = useState(0);

  const refreshStats = useCallback(async () => {
    try {
      const data = await apiFetch<AdminStats>('/admin/stats');
      setCollections(data.collections);
      setTotalDocuments(data.total_documents);
      setTotalFaqs(data.total_faqs);

      // Auto-select collection if none selected
      const docColls = data.collections.filter((c) => c.type === 'documents');
      setCurrentCollection((prev) => {
        if (prev && docColls.some((c) => c.name === prev)) return prev;
        const def = docColls.find((c) => c.name === 'three-stage-search');
        return def?.name || docColls[0]?.name || '';
      });
    } catch (e) {
      console.error('Failed to load stats:', e);
    }
  }, []);

  const login = useCallback(
    async (key: string) => {
      const ok = await validateKey(key);
      if (!ok) throw new Error('Invalid API key');
      setAdminKey(key);
      setKey(key);
      setLoggedIn(true);
      sessionStorage.setItem('qdrant_admin_key', key);
      await refreshStats();
    },
    [refreshStats],
  );

  const logout = useCallback(() => {
    sessionStorage.removeItem('qdrant_admin_key');
    setAdminKey('');
    setKey('');
    setLoggedIn(false);
    setCollections([]);
  }, []);

  const setCollection = useCallback((name: string) => {
    setCurrentCollection(name);
  }, []);

  // Restore session on mount
  useEffect(() => {
    const stored = sessionStorage.getItem('qdrant_admin_key');
    if (stored) {
      login(stored).catch(() => {
        sessionStorage.removeItem('qdrant_admin_key');
      });
    }
  }, [login]);

  const value = useMemo(
    () => ({
      adminKey,
      isLoggedIn,
      currentCollection,
      collections,
      totalDocuments,
      totalFaqs,
      login,
      logout,
      setCollection,
      refreshStats,
    }),
    [adminKey, isLoggedIn, currentCollection, collections, totalDocuments, totalFaqs, login, logout, setCollection, refreshStats],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used inside AppProvider');
  return ctx;
}
