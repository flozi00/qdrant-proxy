/* ============================================================================
 * Layout — Header, tab navigation, and active tab content
 * ============================================================================ */

import { useState } from 'react';
import { collectionId, isPrimarySearchCollection, useApp } from '../store';
import type { TabName } from '../types';
import FaqTab from './FaqTab';
import MaintenanceTab from './MaintenanceTab';
import QualityTab from './QualityTab';
import SearchTab from './SearchTab';

const TABS: { id: TabName; label: string }[] = [
  { id: 'search', label: 'Search' },
  { id: 'faq', label: 'FAQ / KV' },
  { id: 'quality', label: 'Quality Feedback' },
  { id: 'maintenance', label: 'Maintenance' },
];

export default function Layout() {
  const { collections, currentCollection, logout, setCollection } = useApp();
  const [activeTab, setActiveTab] = useState<TabName>('search');
  const documentCollections = collections.filter(isPrimarySearchCollection);

  return (
    <>
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:justify-between lg:items-center">
            <h1 className="text-2xl font-bold text-blue-600">Qdrant Proxy Admin</h1>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              {documentCollections.length > 0 && (
                <label className="flex flex-col gap-1 text-sm text-gray-600 min-w-72">
                  <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                    Active document collection
                  </span>
                  <select
                    value={currentCollection}
                    onChange={(e) => setCollection(e.target.value)}
                    className="border border-gray-300 rounded px-3 py-2 bg-white text-sm focus:ring-2 focus:ring-blue-400"
                  >
                    {documentCollections.map((collection) => {
                      const id = collectionId(collection);
                      return (
                        <option key={id} value={id}>
                          {id} ({collection.count} pts)
                        </option>
                      );
                    })}
                  </select>
                </label>
              )}
              <button onClick={logout} className="text-sm text-red-600 hover:text-red-800 self-start sm:self-auto">
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex overflow-x-auto">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'search' && <SearchTab />}
        {activeTab === 'faq' && <FaqTab />}
        {activeTab === 'quality' && <QualityTab />}
        {activeTab === 'maintenance' && <MaintenanceTab />}
      </div>
    </>
  );
}
