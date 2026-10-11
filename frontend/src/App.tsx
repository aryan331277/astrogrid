import React, { useState, useCallback } from 'react';

type TabId = 'overview' | 'multi-messenger' | 'catalog' | 'settings';

interface Tab {
  id: TabId;
  label: string;
  icon: string;
}

const TABS: Tab[] = [
  { id: 'overview', label: 'Overview', icon: '📊' },
  { id: 'multi-messenger', label: 'Multi-Messenger', icon: '🔭' },
  { id: 'catalog', label: 'Catalog', icon: '📚' },
  { id: 'settings', label: 'Settings', icon: '⚙️' },
];

const OverviewPanel: React.FC = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Observatory Overview</h2>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-400">Active Detectors</h3>
        <p className="text-3xl font-bold mt-2">42</p>
        <p className="text-sm text-gray-400">Across 6 facilities</p>
      </div>
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-green-400">Events Today</h3>
        <p className="text-3xl font-bold mt-2">1,284</p>
        <p className="text-sm text-gray-400">+12% vs yesterday</p>
      </div>
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-purple-400">Data Throughput</h3>
        <p className="text-3xl font-bold mt-2">2.4 TB</p>
        <p className="text-sm text-gray-400">Last 24 hours</p>
      </div>
    </div>
  </div>
);

const MultiMessengerPanel: React.FC = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Multi-Messenger Visualization</h2>
    <div className="bg-gray-800 rounded-lg p-4 h-96 flex items-center justify-center">
      <p className="text-gray-400">Sky map visualization will render here</p>
    </div>
  </div>
);

const CatalogPanel: React.FC = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Event Catalog</h2>
    <div className="bg-gray-800 rounded-lg p-4">
      <table className="w-full text-left">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="py-2 px-4">ID</th>
            <th className="py-2 px-4">Type</th>
            <th className="py-2 px-4">RA</th>
            <th className="py-2 px-4">Dec</th>
            <th className="py-2 px-4">SNR</th>
          </tr>
        </thead>
        <tbody>
          <tr className="border-b border-gray-700">
            <td className="py-2 px-4">GW240101_001</td>
            <td className="py-2 px-4">Binary Black Hole</td>
            <td className="py-2 px-4">12.45°</td>
            <td className="py-2 px-4">-34.21°</td>
            <td className="py-2 px-4">18.7</td>
          </tr>
          <tr className="border-b border-gray-700">
            <td className="py-2 px-4">GRB240102_002</td>
            <td className="py-2 px-4">Gamma-Ray Burst</td>
            <td className="py-2 px-4">234.67°</td>
            <td className="py-2 px-4">12.89°</td>
            <td className="py-2 px-4">9.3</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
);

const SettingsPanel: React.FC = () => (
  <div className="p-6">
    <h2 className="text-2xl font-bold mb-4">Platform Settings</h2>
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-300">API Endpoint</label>
        <input
          type="text"
          defaultValue="http://localhost:8000"
          className="mt-1 block w-full rounded-md border-gray-600 bg-gray-700 text-white"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-300">Refresh Interval (s)</label>
        <input
          type="number"
          defaultValue={30}
          className="mt-1 block w-full rounded-md border-gray-600 bg-gray-700 text-white"
        />
      </div>
    </div>
  </div>
);

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  const handleTabChange = useCallback((tabId: TabId) => {
    setActiveTab(tabId);
  }, []);

  const renderPanel = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPanel />;
      case 'multi-messenger':
        return <MultiMessengerPanel />;
      case 'catalog':
        return <CatalogPanel />;
      case 'settings':
        return <SettingsPanel />;
      default:
        return <OverviewPanel />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <h1 className="text-xl font-bold">AstroGrid</h1>
        <p className="text-sm text-gray-400">Unified Multi-Messenger Observational Astrophysics Platform</p>
      </header>
      <nav className="bg-gray-800 border-b border-gray-700 px-6">
        <div className="flex space-x-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`px-4 py-3 text-sm font-medium rounded-t-lg transition-colors ${
                activeTab === tab.id
                  ? 'bg-gray-900 text-white border-b-2 border-blue-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </nav>
      <main className="flex-1 bg-gray-900">{renderPanel()}</main>
      <footer className="bg-gray-800 border-t border-gray-700 px-6 py-3 text-center text-sm text-gray-500">
        AstroGrid v1.0.0 — © 2024
      </footer>
    </div>
  );
};

export default App;