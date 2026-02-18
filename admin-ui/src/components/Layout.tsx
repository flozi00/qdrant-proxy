/* ============================================================================
 * Layout — Header, tab navigation, and active tab content
 * ============================================================================ */

import { useState } from 'react';
import { useApp } from '../store';
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
  const { logout } = useApp();
  const [activeTab, setActiveTab] = useState<TabName>('search');

  return (
    <>
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-blue-600">Qdrant Proxy Admin</h1>
            <button onClick={logout} className="text-sm text-red-600 hover:text-red-800">
              Logout
            </button>
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
